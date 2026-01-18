import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def test_queue__invalid_filename(self):
    """Ensures queue() correctly handles invalid filenames."""
    with self.assertRaises(pygame.error):
        pygame.mixer.music.queue('')