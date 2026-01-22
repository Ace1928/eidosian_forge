import pygame
import numpy as np

# Initialize Pygame mixer
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1)


# Function to generate tone
def play_tone(frequency, duration, volume=0.5):
    fs = 44100  # Sampling rate, 44100 samples per second
    t = np.linspace(0, duration, int(fs * duration), False)
    tone = volume * np.sin(2 * np.pi * frequency * t)
    sound = np.asarray([tone, tone]).T * 32767 / np.max(np.abs(tone))  # Stereo sound
    sound = sound.astype(np.int16)
    sound_obj = pygame.sndarray.make_sound(sound.copy())
    sound_obj.play(-1)
    pygame.time.delay(int(duration * 1000))
    sound_obj.stop()


# Define a mapping of keyboard keys to frequencies (in Hz)
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

# Set up the display
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Simple Synthesizer")

# Main event loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in key_to_frequency:
                play_tone(key_to_frequency[event.key], duration=0.5)

pygame.quit()
