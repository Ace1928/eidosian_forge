import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_stop__while_playing(self):
    """Ensure stop stops a playing sound."""
    try:
        expected_channels = 0
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        sound = mixer.Sound(file=filename)
        sound.play(-1)
        sound.stop()
        self.assertEqual(sound.get_num_channels(), expected_channels)
    finally:
        pygame.mixer.quit()
        with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
            sound.stop()