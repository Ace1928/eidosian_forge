import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_sound__before_init(self):
    """Ensure exception raised for Sound() creation with non-init mixer."""
    mixer.quit()
    filename = example_path(os.path.join('data', 'house_lo.wav'))
    with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
        mixer.Sound(file=filename)