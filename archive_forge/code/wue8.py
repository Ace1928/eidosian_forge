"""
Module Header

This program generates high-quality sound arrays using specified timbre and harmonics, and saves them to disk as WAV files.
The goal is to be able to create complex sound libraries of notes across octaves for use in a synthesizer or sampler.
The library ideally will contain notes of different timing frequencies, standard volume, all different timbers, different harmonic weights,  and different octaves.
"""

import pygame
import numpy as np
import json
import os
import logging
import threading
import pygame_gui
from typing import Dict, Tuple, Any, List
import wave
from collections import defaultdict

# Setup detailed logging for operational insights and debugging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Pygame mixer with high-quality audio settings for optimal performance
pygame.init()
pygame.mixer.init(frequency=96000, size=-16, channels=2, buffer=512)
logging.info(
    "Pygame mixer initialized with high-quality audio settings for optimal performance."
)

# Constants and configurations
NOTE_FREQUENCIES: Dict[str, float] = {
    "C": 261.63,
    "C#": 277.18,
    "D": 293.66,
    "D#": 311.13,
    "E": 329.63,
    "F": 349.23,
    "F#": 369.99,
    "G": 392.00,
    "G#": 415.30,
    "A": 440.00,
    "A#": 466.16,
    "B": 493.88,
}
OCTAVE_RANGE: range = range(0, 8)
sound_cache: Dict[str, pygame.mixer.Sound] = {}


# Function to generate high-quality sound array using specified timbre and harmonics
def generate_sound_array(
    frequency: float,
    duration: float,
    volume: float,
    harmonics_weights: List[float],
    timbre: str,
) -> np.ndarray:
    fs: int = 96000  # Sampling rate
    t: np.ndarray = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone: np.ndarray = np.zeros_like(t)
    for i, weight in enumerate(harmonics_weights):
        harmonic_frequency: float = (i + 1) * frequency
        if timbre == "sine":
            tone += weight * np.sin(2 * np.pi * harmonic_frequency * t)
        elif timbre == "square":
            tone += weight * np.sign(np.sin(2 * np.pi * harmonic_frequency * t))
        elif timbre == "sawtooth":
            tone += weight * 2 * (t * harmonic_frequency % 1 - 0.5)
    sound_array: np.ndarray = np.asarray([tone] * 2).T * 32767 / np.max(np.abs(tone))
    sound_array = sound_array.astype(np.int16)
    return sound_array


# Function to save sound array to a WAV file
def save_sound_to_file(sound_array: np.ndarray, filename: str) -> None:
    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(96000)
        wf.writeframes(sound_array.tobytes())
        logging.info(f"Saved {filename} to disk with high-quality audio settings.")


# Function to generate and save notes across octaves
def generate_and_save_notes() -> None:
    if not os.path.exists("sounds"):
        os.makedirs("sounds")
    for octave in OCTAVE_RANGE:
        for note, freq in NOTE_FREQUENCIES.items():
            frequency: float = freq * (
                2 ** (octave - 4)
            )  # Middle C (C4) as the reference octave
            filename: str = f"sounds/{note}{octave}.wav"
            if not os.path.exists(filename):
                sound_array: np.ndarray = generate_sound_array(
                    frequency, 1.0, 0.75, [1, 0.5, 0.25], "sine"
                )
                save_sound_to_file(sound_array, filename)
                sound_cache[filename] = pygame.mixer.Sound(filename)
                logging.info(f"Generated and cached {filename}")
