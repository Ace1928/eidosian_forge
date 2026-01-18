import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def test_queue__invalid_sound_type(self):
    """Ensures queue() correctly handles invalid file types."""
    not_a_sound_file = example_path(os.path.join('data', 'city.png'))
    with self.assertRaises(pygame.error):
        pygame.mixer.music.queue(not_a_sound_file)