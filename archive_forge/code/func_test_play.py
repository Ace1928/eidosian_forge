import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def test_play(self):
    filename = example_path(os.path.join('data', 'house_lo.mp3'))
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    self.assertTrue(pygame.mixer.music.get_busy())
    pygame.mixer.music.stop()