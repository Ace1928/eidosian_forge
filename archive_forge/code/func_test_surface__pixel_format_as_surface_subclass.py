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
def test_surface__pixel_format_as_surface_subclass(self):
    """Ensure a subclassed surface can be used for pixel format
        when creating a new surface."""
    expected_depth = 16
    expected_flags = SRCALPHA
    expected_size = (13, 37)
    depth_surface = SurfaceSubclass((11, 21), expected_flags, expected_depth)
    surface = pygame.Surface(expected_size, expected_flags, depth_surface)
    self.assertIsNot(surface, depth_surface)
    self.assertIsInstance(surface, pygame.Surface)
    self.assertNotIsInstance(surface, SurfaceSubclass)
    self.assertEqual(surface.get_size(), expected_size)
    self.assertEqual(surface.get_flags(), expected_flags)
    self.assertEqual(surface.get_bitsize(), expected_depth)