#!/usr/bin/python3

import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


def create_frequency_histogram(
    input_array, number_of_bins, fit_gaussian=False, add_numbers=True
):
    # Frequency Histogram
    num_bins = number_of_bins
    range_min = np.min(input_array)
    range_max = np.max(input_array)
    avg_val = input_array.mean()
    std_val = input_array.std()

    # Create the histogram
    hist, bins = np.histogram(input_array, bins=num_bins, range=(range_min, range_max))

    # # Plot the histogram
    plt.hist(
        input_array,
        bins=num_bins,
        range=(range_min, range_max),
        edgecolor="black",
        alpha=0.7,
    )
    if fit_gaussian:
        # Fit a Gaussian function to the data
        mu, sigma = norm.fit(input_array)
        # Generate the Gaussian function
        x = np.linspace(avg_val - 4 * std_val, avg_val + 4 * std_val, 100)
        p = norm.pdf(x, mu, sigma) * 5.8

        # Plot the Gaussian function
        plt.plot(x, p, color="red", label="Gaussian Function")
    if add_numbers:
        # Add text annotations
        for i, v in enumerate(hist):
            plt.text(
                bins[i] + (bins[i + 1] - bins[i]) / 2,
                v,
                str(v),
                color="black",
                ha="center",
                va="bottom",
            )

    plt.xlabel("Measurement value [m/s²]")
    plt.ylabel("Frequency")
    plt.title("Frequency Histogram")
    plt.show()


def get_frequency_table(input_array):
    # Frequency Histogram
    num_bins = 8
    range_min = np.min(input_array)
    range_max = np.max(input_array)

    # Create the histogram
    hist, bins = np.histogram(input_array, bins=num_bins, range=(range_min, range_max))

    # Calculate bin limits
    bin_limits = [
        f"{start:.6f} ≤ x < {end:.6f}" for start, end in zip(bins[:-1], bins[1:])
    ]

    # # Calculate relative frequency in percentiles
    relative_frequency = hist / len(input_array) * 100

    # # Calculate cumulative relative frequency
    cumulative_relative_frequency = np.cumsum(relative_frequency)

    # # Create a DataFrame to display the frequency table
    frequency_table = pd.DataFrame(
        {
            "Bin Limits": bin_limits,
            "Frequency": hist,
            "Cumulative Frequency": np.cumsum(hist),
            "Relative Frequency (%)": relative_frequency,
            "Cumulative Relative Frequency (%)": cumulative_relative_frequency,
        }
    )
    return frequency_table


def create_metrics_figure(input_array):
    avg_val = input_array.mean()
    med_val = np.median(input_array)
    mode = st.mode(input_array)

    # Plot the ordered sensor values
    x = np.arange(len(sorted_array))
    plt.plot(x, sorted_array, label="Sensor data [m/s²]")
    plt.axhline(avg_val, color="r", linestyle="--", label=f"Mean: {avg_val:.6f} m/s²")
    plt.axhline(
        med_val, color="brown", linestyle="--", label=f"Median: {med_val:.6f} m/s²"
    )
    plt.axhline(mode, color="g", linestyle="--", label=f"Mode: {mode:.6f} m/s²")
    plt.axvline(
        np.where(sorted_array == q1)[0][0],
        color="b",
        linestyle="--",
        label=f"25th percentile: {q1:.6f} m/s²",
    )
    plt.axvline(
        np.where(sorted_array == q3)[0][0],
        color="b",
        linestyle="--",
        label=f"75th percentile: {q3:.6f} m/s²",
    )
    plt.axhline(
        min(sorted_array),
        color="purple",
        linestyle="--",
        label=f"Min: {min(sorted_array):.6f} m/s²",
    )
    plt.axhline(
        max(sorted_array),
        color="orange",
        linestyle="--",
        label=f"Max: {max(sorted_array):.6f} m/s²",
    )
    # Add legend
    plt.legend()

    # Add labels and title
    plt.xlabel("Measurement number")
    plt.ylabel("Measurement value [m/s²]")
    plt.title("Sorted sensor data with metrics")
    plt.show()


if __name__ == "__main__":
    input_arr = np.loadtxt("imu_acc_z.txt", dtype="float")

    sorted_array = np.sort(input_arr)
    input_array = input_arr.copy()

    min_val = input_array.min()
    max_val = input_array.max()
    range = max_val - min_val
    avg_val = input_array.mean()
    var_val = input_array.var()
    std_val = input_array.std()
    med_val = np.median(input_array)
    mode = st.mode(input_array)
    rel_dev = std_val / avg_val

    # Calculate the IQR
    q3, q1 = np.percentile(sorted_array, [75, 25])
    iqr = q3 - q1
    print(f"25th percentile: {q1} m/s², 75th percentile: {q3} m/s²")

    output_string = f"""
    Min Value: {min_val}
    Max Value: {max_val}
    Range: {range}
    IQR: {iqr}
    Average Value: {avg_val}
    Standard Deviation: {std_val}
    Variance: {var_val}
    Relative Standard Deviation: {rel_dev*100}%
    Median Value: {med_val}
    Mode Value: {mode}
    """
    print(output_string)

    create_metrics_figure(input_array)
    create_frequency_histogram(input_array, 8)
    create_frequency_histogram(input_array, 1000, True, False)
    print(get_frequency_table(input_array))
