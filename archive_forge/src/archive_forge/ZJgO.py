import pygame
import numpy as np
import json
from typing import Dict

# Initialize Pygame mixer
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1)

)

# Function to generate tone
def play_tone(frequency: float, duration: float, volume: float = 0.5) -> None:
    """
    Play a tone with the given frequency, duration, and volume.
    
    Args:
        frequency (float): Frequency of the tone in Hz.
        duration (float): Duration of the tone in seconds.
        volume (float): Volume of the tone, default is 0.5.
    """
    fs = 44100  # Sampling rate, 44100 samples per second
    t = np.linspace(0, duration, int(fs * duration), False)
    tone = volume * np.sin(2 * np.pi * frequency * t)
    sound = np.asarray([tone, tone]).T * 32767 / np.max(np.abs(tone))  # Stereo sound
    sound = sound.astype(np.int16)
    sound_obj = pygame.sndarray.make_sound(sound.copy())
    sound_obj.play(-1)
    pygame.time.delay(int(duration * 1000))
    sound_obj.stop()

# Load or define a mapping of keyboard keys to frequencies (in Hz)
def load_or_define_frequencies() -> Dict[int, float]:
    """
    Load the key to frequency mapping from a JSON file or define default values.
    
    Returns:
        Dict[int, float]: Mapping of Pygame key constants to frequencies in Hz.
    """
    try:
        with open("key_frequencies.json", "r") as file:
            key_to_frequency = json.load(file)
    except FileNotFoundError:
        key_to_frequency = {
            pygame.K_a: 261.63,  # C4
            pygame.K_s: 293.66,  # D4
            pygame.K_d: 329.63,  # E4
            pygame.K_f: 349.23,  # F4
            pygame.K_g: 392.00,  # G4
            pygame.K_h: 440.00,  # A4
            pygame.K_j: 493.88,  # B4
            pygame.K_k: 523.25,  # C5
        }
        with open("key_frequencies.json", "w") as file:
            json.dump(key_to_frequency, file)
    return key_to_frequency

key_to_frequency = load_or_define_frequencies()

# Set up the display
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Simple Synthesizer")

# Function to handle configuration changes
def configure_keys() -> None:
    """
    Allow the user to configure the frequency associated with each key.
    """
    print("Press the key you want to configure, or 'q' to quit configuration:")
    configuring = True
    while configuring:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    configuring = False
                else:
                    new_freq = input(
                        f"Enter new frequency for {pygame.key.name(event.key)}: "
                    )
                    if new_freq.isdigit():
                        key_to_frequency[event.key] = float(new_freq)
                        with open("key_frequencies.json", "w") as file:
                            json.dump(key_to_frequency, file)
                    else:
                        print("Invalid frequency input. Please enter a numeric value.")

# Main event loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                configure_keys()
            elif event.key in key_to_frequency:
                play_tone(key_to_frequency[event.key], duration=0.5)

pygame.quit()
