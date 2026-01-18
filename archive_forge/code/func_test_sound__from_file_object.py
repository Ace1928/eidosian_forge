import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_sound__from_file_object(self):
    """Ensure Sound() creation with a file object works."""
    filename = example_path(os.path.join('data', 'house_lo.wav'))
    with open(filename, 'rb') as file_obj:
        sound = mixer.Sound(file_obj)
        self.assertIsInstance(sound, mixer.Sound)