import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft
from sklearn.preprocessing import MinMaxScaler
import threading
import queue
import time
import psutil
import os
import sys
import logging
from datetime import datetime
from scipy.signal import spectrogram, find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import gc
from pydub import AudioSegment
from scipy.io.wavfile import write as wav_write

# Global variables for controlling the visualization
running = True
paused = False
signal_queue = queue.Queue()
default_file_path = "/home/lloyd/Downloads/audio/output.mp3"
output_dir = "/home/lloyd/Downloads/tensorwave/outputs"
os.makedirs(output_dir, exist_ok=True)

# Initialize logging
logging.basicConfig(
    filename="system_resources.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def ensure_system_resources():
    logging.info("Checking system resources...")
    print("Checking system resources...")

    memory_info = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)

    logging.info(
        f"Memory available: {memory_info.available / (1024 ** 2):.2f} MB, CPU usage: {cpu_percent}%"
    )
    print(
        f"Memory available: {memory_info.available / (1024 ** 2):.2f} MB, CPU usage: {cpu_percent}%"
    )

    if memory_info.percent > 95:
        logging.error("RAM usage exceeded 95%. Exiting.")
        print("RAM usage exceeded 95%. Exiting.")
        raise SystemError("Not enough system resources available.")
    else:
        logging.info(f"RAM usage check passed: {memory_info.percent}% used")
        print(f"RAM usage check passed: {memory_info.percent}% used")

    if memory_info.available < memory_info.total * 0.2:
        logging.warning("Low memory detected, optimizing memory usage...")
        print("Low memory detected, optimizing memory usage...")
        global default_file_path
        signal, sr = load_audio(default_file_path)
        signal_queue.put((signal, sr))
        logging.info("Memory optimization complete.")
        print("Memory optimization complete.")


def load_audio(file_path: str = default_file_path) -> tuple[np.ndarray, int]:
    """Load an audio file and return the signal and sample rate."""
    logging.info(f"Reading audio file from {file_path}")
    print(f"Reading audio file from {file_path}")

    # Load the audio file using pydub
    audio = AudioSegment.from_file(file_path)
    sr = audio.frame_rate
    signal = np.array(audio.get_array_of_samples())

    # Convert to mono if the audio is stereo
    if audio.channels > 1:
        signal = signal.reshape((-1, audio.channels)).mean(axis=1)

    logging.info(f"Audio file read complete: {file_path}")
    print(f"Audio file read complete: {file_path}")
    return signal, sr


def record_audio(duration: int = 5, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Record audio from the microphone for a given duration and sample rate and save it."""
    logging.info(f"Recording audio for {duration} seconds at {sr} sample rate")
    print(f"Recording audio for {duration} seconds at {sr} sample rate")

    signal = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float64")
    sd.wait()

    signal = signal.flatten()

    # Save the recorded audio
    timestamp = get_timestamp()
    file_name = f"{timestamp}_recorded_audio.wav"
    file_path = os.path.join(output_dir, file_name)
    wav_write(file_path, sr, signal.astype(np.int16))  # Convert to int16 for saving

    logging.info(f"Recording complete. Audio saved to {file_path}")
    print(f"Recording complete. Audio saved to {file_path}")

    return signal, sr


def get_timestamp() -> str:
    """Get the current timestamp for unique file naming."""
    logging.info("Generating timestamp for file naming.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Timestamp generated: {timestamp}")
    return timestamp


def get_max_freq_components(sr: int) -> int:
    logging.info("Calculating maximum frequency components based on system resources")
    print("Calculating maximum frequency components based on system resources")

    cpu_count = psutil.cpu_count(logical=False)
    memory_info = psutil.virtual_memory()
    max_freq_components = int((sr / 2) * 0.8)  # 80% of the Nyquist frequency

    logging.info(f"CPU count (physical cores): {cpu_count}")
    logging.info(f"Total memory: {memory_info.total / (1024 ** 2):.2f} MB")
    logging.info(f"Available memory: {memory_info.available / (1024 ** 2):.2f} MB")
    logging.info(f"Max frequency components calculated: {max_freq_components}")

    print(f"CPU count (physical cores): {cpu_count}")
    print(f"Total memory: {memory_info.total / (1024 ** 2):.2f} MB")
    print(f"Available memory: {memory_info.available / (1024 ** 2):.2f} MB")
    print(f"Max frequency components calculated: {max_freq_components}")

    return min(max_freq_components, cpu_count * 1000, memory_info.available // (8 * sr))


def compute_spectral_features(signal: np.ndarray, sr: int) -> dict:
    # Compute basic spectral features manually
    spectral_centroid = np.sum(np.arange(len(signal)) * np.abs(signal)) / np.sum(
        np.abs(signal)
    )
    spectral_bandwidth = np.sqrt(
        np.sum((np.arange(len(signal)) - spectral_centroid) ** 2 * np.abs(signal))
        / np.sum(np.abs(signal))
    )
    spectral_rolloff = np.sum(signal) * 0.85
    zero_crossing_rate = ((signal[:-1] * signal[1:]) < 0).sum()

    # Compute MFCCs manually (simplified version)
    mel_filters = np.linspace(0, sr // 2, num=13)
    mfccs = np.log(np.abs(fft(signal)[: len(mel_filters)]) + 1e-10)

    return {
        "spectral_centroid": np.array([spectral_centroid]),
        "spectral_bandwidth": np.array([spectral_bandwidth]),
        "spectral_rolloff": np.array([spectral_rolloff]),
        "zero_crossing_rate": np.array([zero_crossing_rate]),
        "mfccs": mfccs,
    }


def compute_fft_tensor_wave(signal: np.ndarray) -> np.ndarray:
    """Compute FFT tensor wave representation of the input signal."""
    logging.info("Computing FFT of the input signal")
    freq_components = fft(signal)
    magnitude_spectrum = np.abs(freq_components)
    phase_spectrum = np.angle(freq_components)
    logging.info("FFT computation complete")

    # Construct a 2D tensor representation
    tensor_wave = np.vstack([magnitude_spectrum, phase_spectrum])
    logging.info("FFT tensor wave construction complete")

    return tensor_wave


def compute_spectral_features_tensor_wave(signal: np.ndarray, sr: int) -> np.ndarray:
    """Compute spectral features tensor wave representation of the input signal."""
    logging.info("Extracting spectral features")
    spectral_centroid = np.sum(np.arange(len(signal)) * np.abs(signal)) / np.sum(
        np.abs(signal)
    )
    spectral_bandwidth = np.sqrt(
        np.sum((np.arange(len(signal)) - spectral_centroid) ** 2 * np.abs(signal))
        / np.sum(np.abs(signal))
    )
    spectral_rolloff = np.sum(signal) * 0.85
    zero_crossing_rate = ((signal[:-1] * signal[1:]) < 0).sum()
    logging.info("Spectral feature extraction complete")

    # Construct a 2D tensor representation
    tensor_wave = np.vstack(
        [
            np.full(len(signal), spectral_centroid),
            np.full(len(signal), spectral_bandwidth),
            np.full(len(signal), spectral_rolloff),
            np.full(len(signal), zero_crossing_rate),
        ]
    )
    logging.info("Spectral features tensor wave construction complete")

    return tensor_wave


def compute_mfcc_tensor_wave(signal: np.ndarray, sr: int) -> np.ndarray:
    """Compute MFCC tensor wave representation of the input signal."""
    logging.info("Computing MFCCs")
    mel_filters = np.linspace(0, sr // 2, num=13)
    mfccs = np.log(np.abs(fft(signal)[: len(mel_filters)]) + 1e-10)
    logging.info("MFCC computation complete")

    # Construct a 2D tensor representation
    tensor_wave = np.vstack([mfccs])
    logging.info("MFCC tensor wave construction complete")

    return tensor_wave


def create_tensor_wave(signal: np.ndarray, sr: int) -> dict:
    """Create multiple 2D tensor wave representations of the input signal."""
    logging.info("Starting tensor wave creation process")

    try:
        tensor_waves = {
            "fft": compute_fft_tensor_wave(signal),
            "spectral_features": compute_spectral_features_tensor_wave(signal, sr),
            "mfcc": compute_mfcc_tensor_wave(signal, sr),
        }

        # Normalize and scale tensor representations
        logging.info("Normalizing and scaling tensor representations")
        scaler = MinMaxScaler()
        for key in tensor_waves:
            tensor_waves[key] = scaler.fit_transform(tensor_waves[key].T).T
        logging.info("Normalization and scaling complete")

        logging.info("Tensor wave creation process complete")
        return tensor_waves

    except Exception as e:
        logging.error(
            f"An error occurred during tensor wave creation: {e}", exc_info=True
        )
        raise


def save_visualization(
    signal: np.ndarray, tensor_wave_dict: dict, sr: int, title: str, viz_type: str
):
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

    elif viz_type == "tensor_wave_heatmap":
        sns.heatmap(
            tensor_wave_dict["magnitude_spectrum"].T, cmap="viridis", cbar=True, ax=ax
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency Component")
        ax.set_title(f"{title} - Tensor Wave Heatmap")
        save_plot(fig, f"tensor_wave_heatmap_{timestamp}.png")

    elif viz_type == "tensor_wave_3d_surface":
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(
            time, np.arange(tensor_wave_dict["magnitude_spectrum"].shape[0])
        )
        Z = tensor_wave_dict["magnitude_spectrum"]
        ax.plot_surface(X, Y, Z, cmap="viridis")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency Component")
        ax.set_zlabel("Amplitude")
        ax.set_title(f"{title} - 3D Surface Plot of Tensor Wave")
        save_plot(fig, f"tensor_wave_3d_surface_{timestamp}.png")

    elif viz_type == "tensor_wave_pca":
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(tensor_wave_dict["magnitude_spectrum"].T)
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=time, cmap="viridis")
        fig.colorbar(scatter, ax=ax, label="Time [s]")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title(f"{title} - PCA of Tensor Wave")
        save_plot(fig, f"pca_tensor_wave_{timestamp}.png")

    elif viz_type == "tensor_wave_tsne":
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        tsne_result = tsne.fit_transform(tensor_wave_dict["magnitude_spectrum"].T)
        scatter = ax.scatter(
            tsne_result[:, 0], tsne_result[:, 1], c=time, cmap="plasma"
        )
        fig.colorbar(scatter, ax=ax, label="Time [s]")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.set_title(f"{title} - t-SNE of Tensor Wave")
        save_plot(fig, f"tsne_tensor_wave_{timestamp}.png")

    elif viz_type == "tensor_wave_kmeans":
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(tensor_wave_dict["magnitude_spectrum"].T)
        kmeans = KMeans(n_clusters=3)
        kmeans_result = kmeans.fit_predict(tensor_wave_dict["magnitude_spectrum"].T)
        scatter = ax.scatter(
            pca_result[:, 0], pca_result[:, 1], c=kmeans_result, cmap="viridis"
        )
        fig.colorbar(scatter, ax=ax, label="Cluster")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title(f"{title} - K-Means Clustering of Tensor Wave")
        save_plot(fig, f"kmeans_tensor_wave_{timestamp}.png")

    else:
        raise ValueError(f"Unknown visualization type: {viz_type}")

    plt.close(fig)


def main():
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
        signal, sr = load_audio(file_path)
    elif input_option == "mic":
        duration = int(input("Enter the duration for recording (seconds): ").strip())
        signal, sr = record_audio(duration=duration)
    else:
        logging.error("Invalid input option. Exiting.")
        print("Invalid option. Exiting.")
        return

    logging.info("Starting main processing loop.")
    print("Starting main processing loop.")

    try:
        logging.info("Processing signal.")
        print("Processing signal.")
        tensor_wave = create_tensor_wave(signal, sr)
        visualizations = [
            "original_signal",
            "tensor_wave_heatmap",
            "tensor_wave_3d_surface",
            "tensor_wave_pca",
            "tensor_wave_tsne",
            "tensor_wave_kmeans",
        ]

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
                signal,
                tensor_wave,
                sr,
                title="Wave Visualization",
                viz_type=viz_type,
            )
            logging.info(f"Completed visualization: {viz_type}")
            print(f"Completed visualization: {viz_type}")

            # Real-time feedback at maximum granularity
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            logging.info(
                f"Real-time feedback - Memory available: {memory_info.available / (1024 ** 2):.2f} MB, CPU usage: {cpu_percent}%"
            )
            print(
                f"Real-time feedback - Memory available: {memory_info.available / (1024 ** 2):.2f} MB, CPU usage: {cpu_percent}%"
            )
    except Exception as e:
        logging.error(f"An error occurred in the main loop: {e}")
        print(f"An error occurred in the main loop: {e}")
        running = False

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
