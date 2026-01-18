import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_one_sound_playing(self):
    """
        Test that get_busy returns True when one sound is playing.
        """
    sound = pygame.mixer.Sound(example_path('data/house_lo.wav'))
    sound.play()
    time.sleep(0.2)
    self.assertTrue(pygame.mixer.get_busy())
    sound.stop()