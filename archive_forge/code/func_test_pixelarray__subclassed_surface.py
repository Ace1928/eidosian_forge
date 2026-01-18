import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_pixelarray__subclassed_surface(self):
    """Ensure the PixelArray constructor accepts subclassed surfaces."""
    surface = SurfaceSubclass((3, 5), 0, 32)
    pixelarray = pygame.PixelArray(surface)
    self.assertIsInstance(pixelarray, pygame.PixelArray)