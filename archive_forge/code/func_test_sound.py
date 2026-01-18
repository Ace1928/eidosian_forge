import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_sound(self):
    """Ensure Sound() creation with a filename works."""
    filename = example_path(os.path.join('data', 'house_lo.wav'))
    sound1 = mixer.Sound(filename)
    sound2 = mixer.Sound(file=filename)
    self.assertIsInstance(sound1, mixer.Sound)
    self.assertIsInstance(sound2, mixer.Sound)