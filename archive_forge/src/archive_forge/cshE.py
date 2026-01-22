import pygame
import numpy as np
import json
import os
import logging
from typing import Dict, Tuple, Any, List
import wave
from collections import defaultdict

# Setup logging for detailed debugging and operational information
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Pygame mixer with optimal settings for performance and quality
pygame.init()
pygame.mixer.init(
    frequency=44100, size=-16, channels=2, buffer=(1024 * 1024) * 50
)  # 50 MB buffer
logging.info(
    "Pygame mixer initialized with high-quality audio settings and 1 MB buffer."
)

# Constants
NOTE_FREQUENCIES: Dict[str, float] = {
    "C": 16.35,
    "C#": 17.32,
    "D": 18.35,
    "D#": 19.45,
    "E": 20.60,
    "F": 21.83,
    "F#": 23.12,
    "G": 24.50,
    "G#": 25.96,
    "A": 27.50,
    "A#": 29.14,
    "B": 30.87,
}
OCTAVE_RANGE: range = range(0, 8)  # Octaves 0 to 8

# Sound cache dictionary
sound_cache: Dict[str, pygame.mixer.Sound] = {}


# Generate and save sound files for all notes
def generate_and_save_notes() -> None:
    """
    Generate and save sound files for all notes across all octaves specified in OCTAVE_RANGE.
    Each note is saved as a WAV file in the 'sounds' directory.
    """
    for octave in OCTAVE_RANGE:
        for note, base_freq in NOTE_FREQUENCIES.items():
            frequency: float = base_freq * (2**octave)
            filename: str = f"sounds/{note}{octave}.wav"
            if not os.path.exists(filename):
                sound_array: np.ndarray = generate_sound_array(
                    frequency, 1.0
                )  # 1 second duration
                save_sound_to_file(sound_array, filename)
                logging.info(f"Generated and saved {filename}")
            sound_cache[filename] = pygame.mixer.Sound(filename)


def generate_sound_array(
    frequency: float, duration: float, volume: float = 0.75
) -> np.ndarray:
    """
    Generate a sound array using sine wave generation with specified frequency, duration, and volume.
    """
    fs: int = 44100  # Sampling rate
    t: np.ndarray = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone: np.ndarray = volume * np.sin(2 * np.pi * frequency * t)
    sound_array: np.ndarray = np.asarray([tone] * 2).T * 32767 / np.max(np.abs(tone))
    sound_array = sound_array.astype(np.int16)
    return sound_array


def save_sound_to_file(sound_array: np.ndarray, filename: str) -> None:
    """
    Save a sound array to a WAV file.
    """
    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)  # Stereo
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(44100)  # Frame rate
        wf.writeframes(sound_array.tobytes())


# Load or define frequencies
def load_or_define_frequencies() -> Dict[int, float]:
    """
    Attempt to load the key to frequency mapping from a JSON file. If the file is not found or is corrupt,
    generate default values for one octave starting from middle C (C4) and save them to the file.

    Returns:
        Dict[int, float]: Mapping of Pygame key constants to frequencies in Hz.
    """
    path: str = "key_frequencies.json"
    try:
        with open(path, "r") as file:
            frequencies: Dict[int, float] = json.load(file)
            logging.info("Successfully loaded key frequencies from file.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(
            f"Failed to load key frequencies due to {e}. Generating default frequencies."
        )
        frequencies: Dict[int, float] = {
            pygame.K_a + i: NOTE_FREQUENCIES[note] * (2**4)
            for i, note in enumerate("C C# D D# E F F# G G# A A# B".split())
        }
        with open(path, "w") as file:
            json.dump(frequencies, file, indent=4)
            logging.info("Default key frequencies written to file.")
    return frequencies


# Ensure sound files directory exists
if not os.path.exists("sounds"):
    os.makedirs("sounds")
generate_and_save_notes()

key_to_frequency: Dict[int, float] = load_or_define_frequencies()

# Main event loop
screen: pygame.Surface = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Advanced Simple Synthesizer")
running: bool = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            key = event.key
            if key in key_to_frequency:
                filename: str = f"sounds/{key_to_frequency[key]}.wav"
                if filename in sound_cache:
                    sound_obj: pygame.mixer.Sound = sound_cache[filename]
                    sound_obj.play(-1)  # Play sound indefinitely
        elif event.type == pygame.KEYUP:
            key = event.key
            if key in key_to_frequency:
                filename: str = f"sounds/{key_to_frequency[key]}.wav"
                if filename in sound_cache:
                    sound_obj: pygame.mixer.Sound = sound_cache[filename]
                    sound_obj.stop()  # Stop playing sound

pygame.quit()
logging.info("Pygame terminated.")
