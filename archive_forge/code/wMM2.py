import pygame
import numpy as np
import json
from typing import Dict, Tuple, Any
import os
import sys
import logging

# Setup logging for detailed debugging and operational information
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Pygame mixer with optimal settings for performance and quality
pygame.init()
pygame.mixer.init(
    frequency=44100, size=-16, channels=2, buffer=1024
)  # Adjust buffer size if needed

logging.info("Pygame mixer initialized with high-quality audio settings.")

# Pre-generate and cache sounds
sound_cache = {}


def generate_sound(
    frequency: float, duration: float, volume: float = 0.5
) -> pygame.mixer.Sound:
    fs = 44100
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = volume * np.sin(2 * np.pi * frequency * t)
    sound = np.asarray([tone] * 2).T * 32767 / np.max(np.abs(tone))
    sound = sound.astype(np.int16)
    return pygame.sndarray.make_sound(sound)


def get_cached_sound(
    frequency: float, duration: float, volume: float = 0.5
) -> pygame.mixer.Sound:
    key = (frequency, duration, volume)
    if key not in sound_cache:
        if len(sound_cache) >= 100:  # Limit cache size to 100 entries
            sound_cache.pop(next(iter(sound_cache)))  # Remove the first added item
        sound_cache[key] = generate_sound(frequency, duration, volume)
    return sound_cache[key]


# Load or define frequencies
def load_or_define_frequencies() -> Dict[int, float]:
    path = "key_frequencies.json"
    if os.path.exists(path):
        with open(path, "r") as file:
            return json.load(file)
    else:
        default_frequencies = {
            pygame.K_a: 261.63,
            pygame.K_s: 293.66,
            pygame.K_d: 329.63,
            pygame.K_f: 349.23,
            pygame.K_g: 392.00,
            pygame.K_h: 440.00,
            pygame.K_j: 493.88,
            pygame.K_k: 523.25,
        }
        with open(path, "w") as file:
            json.dump(default_frequencies, file, indent=4)
        return default_frequencies


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
                sound_obj = get_cached_sound(key_to_frequency[event.key], 0.5)
                sound_obj.play()

pygame.quit()
logging.info("Pygame terminated.")
