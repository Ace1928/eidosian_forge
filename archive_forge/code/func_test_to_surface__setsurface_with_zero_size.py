from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__setsurface_with_zero_size(self):
    """Ensures zero sized setsurfaces are handled correctly."""
    expected_ref_count = 2
    expected_flag = SRCALPHA
    expected_depth = 32
    expected_color = pygame.Color('white')
    mask_size = (2, 4)
    mask = pygame.mask.Mask(mask_size, fill=True)
    setsurface = pygame.Surface((0, 0), expected_flag, expected_depth)
    to_surface = mask.to_surface(setsurface=setsurface)
    self.assertIsInstance(to_surface, pygame.Surface)
    if not IS_PYPY:
        self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
    self.assertTrue(to_surface.get_flags() & expected_flag)
    self.assertEqual(to_surface.get_bitsize(), expected_depth)
    self.assertEqual(to_surface.get_size(), mask_size)
    assertSurfaceFilled(self, to_surface, expected_color)