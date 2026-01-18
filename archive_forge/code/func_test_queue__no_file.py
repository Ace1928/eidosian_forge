import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def test_queue__no_file(self):
    """Ensures queue() correctly handles missing the file argument."""
    with self.assertRaises(TypeError):
        pygame.mixer.music.queue()