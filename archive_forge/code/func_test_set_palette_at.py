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
def test_set_palette_at(self):
    surf = pygame.Surface((2, 2), 0, 8)
    original = surf.get_palette_at(10)
    replacement = Color(1, 1, 1, 255)
    if replacement == original:
        replacement = Color(2, 2, 2, 255)
    surf.set_palette_at(10, replacement)
    self.assertEqual(surf.get_palette_at(10), replacement)
    next = tuple(original)
    surf.set_palette_at(10, next)
    self.assertEqual(surf.get_palette_at(10), next)
    next = tuple(original)[0:3]
    surf.set_palette_at(10, next)
    self.assertEqual(surf.get_palette_at(10), next)
    self.assertRaises(IndexError, surf.set_palette_at, 256, replacement)
    self.assertRaises(IndexError, surf.set_palette_at, -1, replacement)