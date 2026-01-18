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
def test_unmap_rgb(self):
    surf = pygame.Surface((2, 2), 0, 8)
    c = (1, 1, 1)
    i = 67
    surf.set_palette_at(i, c)
    unmapped_c = surf.unmap_rgb(i)
    self.assertEqual(unmapped_c, c)
    self.assertIsInstance(unmapped_c, pygame.Color)
    c = (128, 64, 12, 255)
    formats = [(0, 16), (0, 24), (0, 32), (SRCALPHA, 16), (SRCALPHA, 32)]
    for flags, bitsize in formats:
        surf = pygame.Surface((2, 2), flags, bitsize)
        unmapped_c = surf.unmap_rgb(surf.map_rgb(c))
        surf.fill(c)
        comparison_c = surf.get_at((0, 0))
        self.assertEqual(unmapped_c, comparison_c, '%s != %s, flags: %i, bitsize: %i' % (unmapped_c, comparison_c, flags, bitsize))
        self.assertIsInstance(unmapped_c, pygame.Color)