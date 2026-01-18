import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_sound__from_pathlib(self):
    """Ensure Sound() creation with a pathlib.Path object works."""
    path = pathlib.Path(example_path(os.path.join('data', 'house_lo.wav')))
    sound1 = mixer.Sound(path)
    sound2 = mixer.Sound(file=path)
    self.assertIsInstance(sound1, mixer.Sound)
    self.assertIsInstance(sound2, mixer.Sound)