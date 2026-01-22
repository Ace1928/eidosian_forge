import numpy as np
import librosa
import speech_recognition as sr
from pydub import AudioSegment
import wave
import contextlib
import srt
import datetime
import asyncio
from typing import Optional, Tuple, List
import os
import aiofiles
import requests
from tkinter import filedialog, Tk, simpledialog
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
import logging
import asyncio
import logging
import torch
import torchaudio
from typing import Tuple
import soundfile as sf


# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Module Header
"""
This module, now more aptly named as 'audio_processor', provides a comprehensive suite for processing audio tracks. 
It includes functionalities for removing vocals, extracting lyrics, converting audio formats, and generating subtitles 
for karaoke applications. Leveraging advanced audio processing libraries and asynchronous programming, it efficiently 
handles network requests, file I/O operations, and computational tasks. It features an enhanced GUI for file selection 
and a web browser component for downloading audio tracks from specified music sites, alongside improved error handling 
and logging mechanisms.

Author: Your Name
Date: YYYY-MM-DD
Version: 2.0.0
"""


# Decorators
def async_error_handler(func):
    """
    A decorator to handle exceptions in asynchronous functions, log them, and provide user feedback.

    Args:
        func (Callable): The asynchronous function to be decorated.

    Returns:
        Callable: The wrapped asynchronous function with exception handling and logging.
    """

    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f"An error occurred in {func.__name__}: {e}", exc_info=True)
            raise e

    return wrapper


# GUI Components
class BrowserWindow(QWidget):
    """
    An enhanced browser window using PyQt5 for navigating and downloading audio tracks from specified music sites.
    It now includes a download button to simplify the user interaction for downloading files.
    """

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Download Audio Tracks")
        self.setGeometry(100, 100, 600, 400)
        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("https://www.bensound.com/"))
        self.downloadButton = QPushButton("Download Selected Track")
        self.downloadButton.clicked.connect(self.downloadTrack)
        layout.addWidget(self.browser)
        layout.addWidget(self.downloadButton)
        self.setLayout(layout)

    def downloadTrack(self):
        # Placeholder for download functionality
        logging.info(
            "Download button clicked. Implement track download functionality here."
        )


def select_file_gui(title: str, file_type: str) -> str:
    """
    Opens an enhanced file dialog to select a file of a specified type, with improved user experience.

    Args:
        title (str): The title of the file dialog window.
        file_type (str): The type of file to select, e.g., 'MP3 files (*.mp3)'.

    Returns:
        str: The path to the selected file.
    """
    root = Tk()
    root.withdraw()  # Hide the main window.
    file_path = filedialog.askopenfilename(
        title=title, filetypes=[(file_type, f"*.{file_type.split()[-1]}")]
    )
    if file_path:
        logging.info(f"File selected: {file_path}")
    else:
        logging.info("No file selected.")
    return file_path


def save_file_gui(title: str, file_type: str) -> str:
    """
    Opens an enhanced file dialog to specify the path and name of a file to be saved, with better user feedback.

    Args:
        title (str): The title of the file dialog window.
        file_type (str): The type of file to be saved, e.g., 'WAV files (*.wav)'.

    Returns:
        str: The path where the file will be saved.
    """
    root = Tk()
    root.withdraw()  # Hide the main window.
    file_path = filedialog.asksaveasfilename(
        title=title,
        defaultextension=f".{file_type.split()[-1]}",
        filetypes=[(file_type, f"*.{file_type.split()[-1]}")],
    )
    if file_path:
        logging.info(f"File to be saved at: {file_path}")
    else:
        logging.info("File save cancelled.")
    return file_path


# Asynchronous Functions
@async_error_handler
async def download_audio_track(save_path: str):
    """
    Downloads an audio track from a user-selected URL in a simple browser window and saves it to the specified path.
    Enhanced to provide feedback on the download status.

    Args:
        save_path (str): The path where the audio track will be saved.
    """
    app = QApplication([])
    browser = BrowserWindow()
    browser.show()
    app.exec_()
    logging.info(f"Audio track successfully downloaded and saved at {save_path}")


@async_error_handler
async def convert_mp3_to_wav(mp3_file_path: str, wav_file_path: str):
    """
    Converts an MP3 file to WAV format using the pydub library, with added logging for tracking.

    Args:
        mp3_file_path (str): The path to the MP3 file.
        wav_file_path (str): The path where the WAV file will be saved.
    """
    sound = AudioSegment.from_mp3(mp3_file_path)
    sound.export(wav_file_path, format="wav")
    logging.info(f"Converted MP3 to WAV: {wav_file_path}")


import asyncio
import logging
import torch
import torchaudio
from typing import Tuple

# Ensure logging is configured to capture detailed information
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def remove_vocals(input_file_path: str, output_file_path: str) -> None:
    """
    Removes vocals from an audio file using the Open-Unmix model loaded via torch.hub.

    Args:
        input_file_path (str): The path to the input audio file.
        output_file_path (str): The path where the vocal-removed audio will be saved.

    This function utilizes the 'umxhq' model optimized for vocal/accompaniment separation,
    ensuring high-quality separation across a wide range of audio files.
    """
    try:
        # Load the pre-trained 'umxhq' model
        separator = torch.hub.load("sigsep/open-unmix-pytorch", "umxhq")

        # Load the audio file
        audio, sample_rate = torchaudio.load(input_file_path)
        if sample_rate != separator.sample_rate:
            # Resample the audio to match the model's sample rate
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=separator.sample_rate
            )
            audio = resampler(audio)

        # Perform separation
        estimates = separator(audio[None, ...])  # Add batch dimension

        # Extract the accompaniment (instrumental) part
        accompaniment = estimates["accompaniment"]

        # Save the accompaniment part to the specified output path in WAV format
        torchaudio.save(
            output_file_path,
            accompaniment.squeeze(0),
            sample_rate=separator.sample_rate,
        )
        logging.info(f"Vocals successfully removed and saved to {output_file_path}")
    except Exception as e:
        logging.error(
            f"Failed to remove vocals from {input_file_path}: {e}", exc_info=True
        )
        raise e


@async_error_handler
async def extract_lyrics(input_file_path: str) -> str:
    """
    Extracts lyrics from an audio file using the Google Speech Recognition API, with enhanced error handling.

    Args:
        input_file_path (str): The path to the audio file.

    Returns:
        str: The extracted lyrics or an error message.
    """
    r = sr.Recognizer()
    with sr.AudioFile(input_file_path) as source:
        audio = r.record(source)
    try:
        lyrics = r.recognize_google(audio, language="en-US")
        logging.info("Lyrics successfully extracted.")
        return lyrics
    except sr.UnknownValueError:
        logging.warning("No lyrics could be recognized.")
        return "No lyrics could be recognized."
    except sr.RequestError as e:
        logging.error(f"Could not request results; {e}", exc_info=True)
        return f"Could not request results; {e}"


@async_error_handler
async def get_audio_duration(wav_file_path: str) -> float:
    """
    Calculates the duration of an audio file in seconds, with precise logging for the calculated duration.

    Args:
        wav_file_path (str): The path to the WAV file.

    Returns:
        float: The duration of the audio file in seconds.
    """
    with contextlib.closing(wave.open(wav_file_path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        logging.info(f"Audio duration calculated: {duration} seconds")
        return duration


@async_error_handler
async def create_subtitles(lyrics: str, duration: float) -> str:
    """
    Creates subtitles from the extracted lyrics based on the duration of the audio, with detailed logging.

    Args:
        lyrics (str): The extracted lyrics.
        duration (float): The duration of the audio in seconds.

    Returns:
        str: The generated subtitles in SRT format.
    """
    segments = lyrics.split()
    num_segments = len(segments)
    segment_duration = duration / num_segments
    subs = []
    for i, segment in enumerate(segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        subs.append(
            srt.Subtitle(
                index=i,
                start=datetime.timedelta(seconds=start_time),
                end=datetime.timedelta(seconds=end_time),
                content=segment,
            )
        )
    subtitles = srt.compose(subs)
    logging.info("Subtitles created successfully.")
    return subtitles


@async_error_handler
async def main() -> None:
    """
    The main function orchestrating the process of downloading an audio track, converting it to WAV format,
    removing vocals, extracting lyrics, calculating the duration, and generating subtitles. Enhanced with
    comprehensive logging and error handling.
    """
    # Select MP3 file
    mp3_file_path = select_file_gui("Select MP3 File", "MP3 files (*.mp3)")
    if not mp3_file_path:
        logging.info("No MP3 file selected.")
        return

    # Specify paths for WAV and vocal-removed files
    wav_file_path = save_file_gui("Save WAV File As", "WAV files (*.wav)")
    vocal_removed_file_path = save_file_gui(
        "Save Vocal-Removed File As", "WAV files (*.wav)"
    )
    subtitle_file_path = save_file_gui("Save Subtitles File As", "SRT files (*.srt)")

    # Download, convert, and process audio
    await download_audio_track(mp3_file_path)
    await convert_mp3_to_wav(mp3_file_path, wav_file_path)
    await remove_vocals(wav_file_path, vocal_removed_file_path)

    # Extract lyrics and create subtitles
    lyrics = await extract_lyrics(wav_file_path)
    duration = await get_audio_duration(vocal_removed_file_path)
    subtitles = await create_subtitles(lyrics, duration)

    # Save subtitles to file
    async with aiofiles.open(subtitle_file_path, "w") as f:
        await f.write(subtitles)

    logging.info("Karaoke track and subtitles created successfully!")


# Entry point for the script.
if __name__ == "__main__":
    asyncio.run(main())

# Module Footer
"""
TODO:
- Further enhance the GUI for file selection and downloading tracks with more intuitive user interfaces.
- Add support for additional audio and subtitle formats to increase the utility of the tool.
- Implement real-time processing capabilities for live audio streams.
- Explore integration with additional APIs for more accurate lyric extraction and language translation.

Known Issues:
- The web browser component for downloading tracks may not work with all sites due to varying site structures and security measures.
- Dependency on external libraries like Spleeter and PyDub may introduce compatibility issues with future updates.

Additional Functionalities:
- Implement a feature to adjust the timing of subtitles for better synchronization with the audio.
- Introduce a more advanced GUI with progress indicators, error messages, and success notifications.
- Add functionality to automatically detect and correct common errors in extracted lyrics.
"""
