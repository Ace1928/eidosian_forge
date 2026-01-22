import pygame
import numpy as np
import json
import os
import logging
from typing import Dict, Tuple, Any

# Setup logging for detailed debugging and operational information
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Pygame mixer with optimal settings for performance and quality
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
logging.info("Pygame mixer initialized with high-quality audio settings.")

# Constants
NOTE_FREQUENCIES = {
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
OCTAVE_RANGE = range(0, 9)  # Octaves 0 to 8


# Generate and save sound files for all notes
def generate_and_save_notes() -> None:
    """
    Generate and save sound files for all notes across all octaves specified in OCTAVE_RANGE.
    Each note is saved as a WAV file in the 'sounds' directory.
    """
    for octave in OCTAVE_RANGE:
        for note, base_freq in NOTE_FREQUENCIES.items():
            frequency = base_freq * (2**octave)
            filename = f"sounds/{note}{octave}.wav"
            if not os.path.exists(filename):
                sound = generate_sound(frequency, 1.0)  # 1 second duration
                pygame.mixer.Sound.save(sound, filename)
                logging.info(f"Generated and saved {filename}")


def generate_sound(
    frequency: float, duration: float, volume: float = 0.5
) -> pygame.mixer.Sound:
    """
    Generate a sound using sine wave generation with specified frequency, duration, and volume.
    Utilizes numpy for efficient numerical computations to ensure high fidelity audio production.

    Args:
        frequency (float): Frequency of the tone in Hz.
        duration (float): Duration of the tone in seconds.
        volume (float): Volume of the tone, default is 0.5.

    Returns:
        pygame.mixer.Sound: A Pygame Sound object created from the generated tone.
    """
    fs = 44100  # Optimal sampling rate for audio quality
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = volume * np.sin(2 * np.pi * frequency * t)
    sound = np.asarray([tone] * 2).T * 32767 / np.max(np.abs(tone))
    sound = sound.astype(np.int16)
    return pygame.sndarray.make_sound(sound)


def get_cached_sound(
    frequency: float, duration: float, volume: float = 0.5
) -> pygame.mixer.Sound:
    """
    Retrieve a sound from the cache or generate it if not present. Manages cache size to limit memory usage.

    Args:
        frequency (float): Frequency of the tone in Hz.
        duration (float): Duration of the tone in seconds.
        volume (float): Volume of the tone, default is 0.5.

    Returns:
        pygame.mixer.Sound: A Pygame Sound object either retrieved from cache or newly generated.
    """
    key = (frequency, duration, volume)
    if key not in sound_cache:
        if len(sound_cache) >= 100:  # Limit cache size to 100 entries
            sound_cache.pop(next(iter(sound_cache)))  # Remove the first added item
        sound_cache[key] = generate_sound(frequency, duration, volume)
    return sound_cache[key]


# Load or define frequencies
def load_or_define_frequencies() -> Dict[int, float]:
    """
    Load the key to frequency mapping from a JSON file or define default values if file not found.
    The mapping is for one octave starting from middle C (C4).

    Returns:
        Dict[int, float]: Mapping of Pygame key constants to frequencies in Hz.
    """
    path = "key_frequencies.json"
    if os.path.exists(path):
        with open(path, "r") as file:
            return json.load(file)
    else:
        # Default mapping for one octave starting from middle C (C4)
        default_frequencies = {
            pygame.K_a + i: NOTE_FREQUENCIES[note] * (2**4)
            for i, note in enumerate(NOTE_FREQUENCIES)
        }
        with open(path, "w") as file:
            json.dump(default_frequencies, file, indent=4)
        return default_frequencies


# Ensure sound files directory exists
if not os.path.exists("sounds"):
    os.makedirs("sounds")
generate_and_save_notes()

key_to_frequency = load_or_define_frequencies()

# Main event loop
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Advanced Simple Synthesizer")
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in key_to_frequency:
                filename = f"sounds/{key_to_frequency[event.key]}.wav"
                sound_obj = pygame.mixer.Sound(filename)
                sound_obj.play()

pygame.quit()
logging.info("Pygame terminated.")
