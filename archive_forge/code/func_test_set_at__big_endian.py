import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
def test_set_at__big_endian(self):
    """png files are loaded in big endian format (BGR rather than RGB)"""
    pygame.display.init()
    try:
        image = pygame.image.load(example_path(os.path.join('data', 'BGR.png')))
        self.assertEqual(image.get_at((10, 10)), pygame.Color(255, 0, 0))
        self.assertEqual(image.get_at((10, 20)), pygame.Color(0, 255, 0))
        self.assertEqual(image.get_at((10, 40)), pygame.Color(0, 0, 255))
        image.set_at((10, 10), pygame.Color(255, 0, 0))
        image.set_at((10, 20), pygame.Color(0, 255, 0))
        image.set_at((10, 40), pygame.Color(0, 0, 255))
        self.assertEqual(image.get_at((10, 10)), pygame.Color(255, 0, 0))
        self.assertEqual(image.get_at((10, 20)), pygame.Color(0, 255, 0))
        self.assertEqual(image.get_at((10, 40)), pygame.Color(0, 0, 255))
    finally:
        pygame.display.quit()