import numpy as np
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft, ifft
from sklearn.preprocessing import MinMaxScaler
import threading
import queue
import time
import psutil
import os
import sys
import select
import librosa.display
import seaborn as sns
import pandas as pd
from datetime import datetime
from scipy.signal import spectrogram, find_peaks
from scipy.stats import kurtosis, skew
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import gc

# Global variables for controlling the visualization
running = True
paused = False
signal_queue = queue.Queue()
default_file_path = "/home/lloyd/Downloads/audio/output.mp3"
output_dir = "/home/lloyd/Downloads/tensorwave/outputs"
os.makedirs(output_dir, exist_ok=True)

import logging

# Initialize logging
logging.basicConfig(
    filename="system_resources.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def ensure_system_resources():
    memory_info = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(
        interval=0.1
    )  # Reduced interval for quicker response

    # Log memory and CPU usage
    logging.info(
        f"Memory available: {memory_info.available / (1024 ** 2):.2f} MB, CPU usage: {cpu_percent}%"
    )
    print(
        f"Memory available: {memory_info.available / (1024 ** 2):.2f} MB, CPU usage: {cpu_percent}%"
    )

    # Check for memory usage
    if memory_info.percent > 95:
        logging.error("RAM usage exceeded 95%. Exiting.")
        print("RAM usage exceeded 95%. Exiting.")
        raise SystemError("Not enough system resources available.")
    else:
        logging.info(f"RAM usage check passed: {memory_info.percent}% used")
        print(f"RAM usage check passed: {memory_info.percent}% used")

    # Dynamic adjustment based on system load
    if memory_info.available < memory_info.total * 0.2:
        logging.warning("Low memory detected, optimizing memory usage...")
        print("Low memory detected, optimizing memory usage...")
        # Implement logic to optimize memory usage, e.g., clear caches, use memory-mapped files, etc.
        # Example: Use memory-mapped files for large data
        global default_file_path
        signal, sr = librosa.load(default_file_path, sr=None, mmap=True)
        signal_queue.put((signal, sr))
        logging.info("Memory optimization complete.")
        print("Memory optimization complete.")


def get_timestamp() -> str:
    """Get the current timestamp for unique file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_audio_file(file_path: str = default_file_path) -> tuple[np.ndarray, int]:
    """Read an audio file and return the signal and sample rate."""
    logging.info(f"Reading audio file from {file_path}")
    signal, sr = librosa.load(file_path, sr=None)
    logging.info(f"Audio file read complete: {file_path}")
    return signal, sr


def record_audio(duration: int = 5, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Record audio from the microphone for a given duration and sample rate."""
    print("Recording...")
    logging.info(f"Recording audio for {duration} seconds at {sr} sample rate")
    signal = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float64")
    sd.wait()
    print("Recording complete.")
    logging.info("Recording complete.")
    return signal.flatten(), sr


def get_max_freq_components(sr: int) -> int:
    """Calculate the maximum frequency components based on system resources."""
    cpu_count = psutil.cpu_count(logical=False)
    memory_info = psutil.virtual_memory()
    max_freq_components = int((sr / 2) * 0.8)  # 80% of the Nyquist frequency
    logging.info(f"Max frequency components calculated: {max_freq_components}")
    return min(max_freq_components, cpu_count * 1000, memory_info.available // (8 * sr))


def create_tensor_wave(signal: np.ndarray, sr: int) -> dict:
    """
    Create a tensor representation of the wave using advanced spectral and frequency analysis.

    This function performs the following steps:
    1. Computes the FFT of the input signal.
    2. Extracts additional spectral features such as spectral centroid, bandwidth, roll-off, and MFCCs.
    3. Constructs a dictionary of tensor representations using these features.
    4. Normalizes and scales each tensor for better interpretability.

    Parameters:
    signal (np.ndarray): The input audio signal.
    sr (int): The sample rate of the audio signal.

    Returns:
    dict: A dictionary of normalized and scaled tensor representations of the wave.
    """
    logging.info("Starting tensor wave creation process")

    # Step 1: Compute the FFT of the input signal
    logging.info("Computing FFT of the input signal")
    freq_components = np.fft.fft(signal)
    magnitude_spectrum = np.abs(freq_components)
    phase_spectrum = np.angle(freq_components)
    logging.info("FFT computation complete")

    # Step 2: Extract additional spectral features
    logging.info("Extracting additional spectral features")
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    logging.info("Spectral feature extraction complete")

    # Step 3: Construct the dictionary of tensor representations
    logging.info("Constructing tensor representation dictionary")
    tensor_wave_dict = {
        "magnitude_spectrum": magnitude_spectrum,
        "phase_spectrum": phase_spectrum,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff": spectral_rolloff,
        "zero_crossing_rate": zero_crossing_rate,
        "mfccs": mfccs,
    }
    logging.info("Tensor representation dictionary construction complete")

    # Step 4: Normalize and scale each tensor
    logging.info("Normalizing and scaling tensor representations")
    scaler = MinMaxScaler()
    for key in tensor_wave_dict:
        logging.info(f"Normalizing and scaling {key}")
        if tensor_wave_dict[key].ndim == 1:
            tensor_wave_dict[key] = tensor_wave_dict[key].reshape(-1, 1)
        tensor_wave_dict[key] = scaler.fit_transform(tensor_wave_dict[key].T).T
        logging.info(f"{key} normalization and scaling complete")

    logging.info("Tensor wave creation process complete")
    return tensor_wave_dict


def save_visualization(
    signal: np.ndarray, tensor_wave_dict: dict, sr: int, title: str, viz_type: str
):
    """Save a specific visualization of the signal and tensor wave."""
    time = np.linspace(0, len(signal) / sr, num=len(signal))
    timestamp = get_timestamp()

    def save_plot(fig, filename):
        fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Saved plot: {filename}")

    fig, ax = plt.subplots(figsize=(10, 4))

    if viz_type == "original_signal":
        ax.plot(time, signal, label="Original Signal")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title} - Original Signal")
        ax.legend()
        save_plot(fig, f"original_signal_{timestamp}.png")

    elif viz_type == "tensor_wave_dict":
        sns.heatmap(tensor_wave_dict[viz_type].T, cmap="viridis", cbar=True, ax=ax)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency Component")
        ax.set_title(f"{title} - Tensor Wave Heatmap")
        save_plot(fig, f"tensor_wave_heatmap_{timestamp}.png")

    elif viz_type == "spectrogram":
        f, t, Sxx = spectrogram(signal, sr)
        cax = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(f"{title} - Spectrogram")
        fig.colorbar(cax, ax=ax, label="Intensity [dB]")
        save_plot(fig, f"spectrogram_{timestamp}.png")

    elif viz_type == "power_spectrum":
        power_spectrum = np.abs(fft(signal)) ** 2
        freqs = np.fft.fftfreq(len(signal), 1 / sr)
        ax.plot(freqs[: len(freqs) // 2], power_spectrum[: len(power_spectrum) // 2])
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power")
        ax.set_title(f"{title} - Power Spectrum")
        save_plot(fig, f"power_spectrum_{timestamp}.png")

    elif viz_type == "zero_crossing_rate":
        zero_crossings = librosa.zero_crossings(signal, pad=False)
        ax.plot(time, signal, label="Signal")
        ax.vlines(
            time[zero_crossings],
            ymin=min(signal),
            ymax=max(signal),
            color="r",
            alpha=0.5,
            label="Zero Crossings",
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title} - Zero Crossing Rate")
        ax.legend()
        save_plot(fig, f"zero_crossing_rate_{timestamp}.png")

    elif viz_type == "rms_energy":
        rms = librosa.feature.rms(y=signal)[0]
        ax.plot(time, signal, label="Signal")
        ax.plot(time[: len(rms)], rms, label="RMS Energy", color="r")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude / RMS Energy")
        ax.set_title(f"{title} - RMS Energy")
        ax.legend()
        save_plot(fig, f"rms_energy_{timestamp}.png")

    elif viz_type == "mel_spectrogram":
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        librosa.display.specshow(
            mel_spectrogram_db, sr=sr, x_axis="time", y_axis="mel", ax=ax
        )
        fig.colorbar(ax.images[0], ax=ax, format="%+2.0f dB")
        ax.set_title(f"{title} - Mel Spectrogram")
        save_plot(fig, f"mel_spectrogram_{timestamp}.png")

    elif viz_type == "chromagram":
        chromagram = librosa.feature.chroma_stft(y=signal, sr=sr)
        librosa.display.specshow(
            chromagram, x_axis="time", y_axis="chroma", cmap="coolwarm", ax=ax
        )
        fig.colorbar(ax.images[0], ax=ax)
        ax.set_title(f"{title} - Chromagram")
        save_plot(fig, f"chromagram_{timestamp}.png")

    elif viz_type == "tonnetz":
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr)
        librosa.display.specshow(tonnetz, x_axis="time", ax=ax)
        fig.colorbar(ax.images[0], ax=ax)
        ax.set_title(f"{title} - Tonnetz")
        save_plot(fig, f"tonnetz_{timestamp}.png")

    elif viz_type == "tempogram":
        tempogram = librosa.feature.tempogram(y=signal, sr=sr)
        librosa.display.specshow(tempogram, sr=sr, x_axis="time", y_axis="tempo", ax=ax)
        fig.colorbar(ax.images[0], ax=ax)
        ax.set_title(f"{title} - Tempogram")
        save_plot(fig, f"tempogram_{timestamp}.png")

    elif viz_type == "spectral_centroid":
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
        ax.semilogy(time, spectral_centroid, label="Spectral Centroid")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Hz")
        ax.set_title(f"{title} - Spectral Centroid")
        ax.legend()
        save_plot(fig, f"spectral_centroid_{timestamp}.png")

    elif viz_type == "spectral_bandwidth":
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
        ax.semilogy(time, spectral_bandwidth, label="Spectral Bandwidth")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Hz")
        ax.set_title(f"{title} - Spectral Bandwidth")
        ax.legend()
        save_plot(fig, f"spectral_bandwidth_{timestamp}.png")

    elif viz_type == "spectral_contrast":
        spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
        librosa.display.specshow(spectral_contrast, x_axis="time", ax=ax)
        fig.colorbar(ax.images[0], ax=ax)
        ax.set_title(f"{title} - Spectral Contrast")
        save_plot(fig, f"spectral_contrast_{timestamp}.png")

    elif viz_type == "spectral_flatness":
        spectral_flatness = librosa.feature.spectral_flatness(y=signal)[0]
        ax.plot(time, spectral_flatness, label="Spectral Flatness")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Flatness")
        ax.set_title(f"{title} - Spectral Flatness")
        ax.legend()
        save_plot(fig, f"spectral_flatness_{timestamp}.png")

    elif viz_type == "kurtosis":
        signal_kurtosis = kurtosis(signal)
        ax.plot(time, signal, label="Signal")
        ax.axhline(y=signal_kurtosis, color="r", linestyle="-", label="Kurtosis")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title} - Kurtosis")
        ax.legend()
        save_plot(fig, f"kurtosis_{timestamp}.png")

    elif viz_type == "skewness":
        signal_skewness = skew(signal)
        ax.plot(time, signal, label="Signal")
        ax.axhline(y=signal_skewness, color="r", linestyle="-", label="Skewness")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title} - Skewness")
        ax.legend()
        save_plot(fig, f"skewness_{timestamp}.png")

    elif viz_type == "peak_detection":
        peaks, _ = find_peaks(signal, height=0)
        ax.plot(time, signal, label="Signal")
        ax.plot(time[peaks], signal[peaks], "x", label="Peaks")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title} - Peak Detection")
        ax.legend()
        save_plot(fig, f"peak_detection_{timestamp}.png")

    elif viz_type == "smoothed_signal":
        smoothed_signal = gaussian_filter1d(signal, sigma=2)
        ax.plot(time, smoothed_signal, label="Smoothed Signal")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title} - Smoothed Signal")
        ax.legend()
        save_plot(fig, f"smoothed_signal_{timestamp}.png")

    elif viz_type == "pca":
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(tensor_wave_dict[viz_type])
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=time, cmap="viridis")
        fig.colorbar(scatter, ax=ax, label="Time [s]")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title(f"{title} - PCA of Tensor Wave")
        save_plot(fig, f"pca_tensor_wave_{timestamp}.png")

    elif viz_type == "tsne":
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        tsne_result = tsne.fit_transform(tensor_wave_dict[viz_type])
        scatter = ax.scatter(
            tsne_result[:, 0], tsne_result[:, 1], c=time, cmap="plasma"
        )
        fig.colorbar(scatter, ax=ax, label="Time [s]")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.set_title(f"{title} - t-SNE of Tensor Wave")
        save_plot(fig, f"tsne_tensor_wave_{timestamp}.png")

    elif viz_type == "kmeans":
        kmeans = KMeans(n_clusters=3)
        kmeans_result = kmeans.fit_predict(tensor_wave_dict[viz_type])
        scatter = ax.scatter(
            pca_result[:, 0], pca_result[:, 1], c=kmeans_result, cmap="viridis"
        )
        fig.colorbar(scatter, ax=ax, label="Cluster")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title(f"{title} - K-Means Clustering of Tensor Wave")
        save_plot(fig, f"kmeans_tensor_wave_{timestamp}.png")

    elif viz_type == "plotly_3d_scatter":
        fig = px.scatter_3d(
            x=pca_result[:, 0], y=pca_result[:, 1], z=tsne_result[:, 0], color=time
        )
        fig.update_layout(
            title=f"{title} - 3D Scatter Plot",
            scene=dict(
                xaxis_title="PCA Component 1",
                yaxis_title="PCA Component 2",
                zaxis_title="t-SNE Component 1",
            ),
        )
        fig.write_html(os.path.join(output_dir, f"3d_scatter_{timestamp}.html"))

    elif viz_type == "plotly_line":
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=time, y=signal, mode="lines", name="Original Signal")
        )
        fig.update_layout(
            title=f"{title} - Original Signal",
            xaxis_title="Time [s]",
            yaxis_title="Amplitude",
        )
        fig.write_html(os.path.join(output_dir, f"original_signal_{timestamp}.html"))

    elif viz_type == "plotly_heatmap":
        fig = go.Figure(
            data=go.Heatmap(
                z=tensor_wave_dict[viz_type].T, x=time, colorscale="Viridis"
            )
        )
        fig.update_layout(
            title=f"{title} - Tensor Wave Heatmap",
            xaxis_title="Time [s]",
            yaxis_title="Frequency Component",
        )
        fig.write_html(
            os.path.join(output_dir, f"tensor_wave_heatmap_{timestamp}.html")
        )

    elif viz_type == "plotly_spectrogram":
        fig = go.Figure(
            data=go.Heatmap(z=10 * np.log10(Sxx), x=t, y=f, colorscale="Viridis")
        )
        fig.update_layout(
            title=f"{title} - Spectrogram",
            xaxis_title="Time [s]",
            yaxis_title="Frequency [Hz]",
        )
        fig.write_html(os.path.join(output_dir, f"spectrogram_{timestamp}.html"))

    elif viz_type == "plotly_3d_surface":
        fig = go.Figure(data=[go.Surface(z=tensor_wave_dict[viz_type].T)])
        fig.update_layout(
            title=f"{title} - 3D Surface Plot of Tensor Wave",
            scene=dict(
                xaxis_title="Time [s]",
                yaxis_title="Frequency Component",
                zaxis_title="Amplitude",
            ),
        )
        fig.write_html(os.path.join(output_dir, f"3d_surface_{timestamp}.html"))

    else:
        raise ValueError(f"Unknown visualization type: {viz_type}")

    plt.close(fig)


def main():
    """Main function to execute the program."""
    global running, paused

    logging.info("Program started.")
    print("Program started.")

    input_option = (
        input("Enter 'file' to load an audio file or 'mic' to record audio: ")
        .strip()
        .lower()
    )
    if input_option == "file":
        file_path = (
            input(
                f"Enter the path to the audio file (default: {default_file_path}): "
            ).strip()
            or default_file_path
        )
        signal, sr = read_audio_file(file_path)
    elif input_option == "mic":
        duration = int(input("Enter the duration for recording (seconds): ").strip())
        signal, sr = record_audio(duration=duration)
    else:
        logging.error("Invalid input option. Exiting.")
        print("Invalid option. Exiting.")
        return

    logging.info("Starting main processing loop.")
    print("Starting main processing loop.")

    while running:
        if not signal_queue.empty():
            chunk = signal_queue.get()
            logging.info("Processing new chunk from signal queue.")
            print("Processing new chunk from signal queue.")
            tensor_wave = create_tensor_wave(chunk, sr)
            visualizations = [
                "original_signal",
                "tensor_wave_heatmap",
                "spectrogram",
                "power_spectrum",
                "zero_crossing_rate",
                "rms_energy",
                "mel_spectrogram",
                "chromagram",
                "tonnetz",
                "tempogram",
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_contrast",
                "spectral_flatness",
                "kurtosis",
                "skewness",
                "peak_detection",
                "smoothed_signal",
                "pca",
                "tsne",
                "kmeans",
                "plotly_3d_scatter",
                "plotly_line",
                "plotly_heatmap",
                "plotly_spectrogram",
                "plotly_3d_surface",
                # Additional visualizations omitted for brevity
            ]

            # Process visualizations one at a time to avoid overloading the system
            for viz_type in visualizations:
                if not running:
                    logging.info(
                        "Stopping visualization processing as running flag is set to False."
                    )
                    print(
                        "Stopping visualization processing as running flag is set to False."
                    )
                    break
                logging.info(f"Generating visualization: {viz_type}")
                print(f"Generating visualization: {viz_type}")
                save_visualization(
                    chunk,
                    tensor_wave,
                    sr,
                    title="Wave Visualization",
                    viz_type=viz_type,
                )
                ensure_system_resources()
                logging.info(f"Completed visualization: {viz_type}")
                print(f"Completed visualization: {viz_type}")

    logging.info("Main processing loop ended.")
    print("Main processing loop ended.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
    finally:
        logging.info("Program terminated.")
        print("Program terminated.")
