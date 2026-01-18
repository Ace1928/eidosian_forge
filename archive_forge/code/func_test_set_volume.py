import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def test_set_volume(self):
    filename = example_path(os.path.join('data', 'house_lo.mp3'))
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    pygame.mixer.music.set_volume(0.5)
    vol = pygame.mixer.music.get_volume()
    self.assertEqual(vol, 0.5)
    pygame.mixer.music.stop()