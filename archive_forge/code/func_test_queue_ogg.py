import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def test_queue_ogg(self):
    """Ensures queue() accepts ogg files.

        |tags:music|
        """
    filename = example_path(os.path.join('data', 'house_lo.ogg'))
    pygame.mixer.music.queue(filename)