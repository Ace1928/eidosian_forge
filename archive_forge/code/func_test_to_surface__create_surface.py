from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__create_surface(self):
    """Ensures empty and full masks can be drawn onto a created surface."""
    expected_ref_count = 2
    expected_flag = SRCALPHA
    expected_depth = 32
    size = (33, 65)
    test_fills = ((pygame.Color('white'), True), (pygame.Color('black'), False))
    for expected_color, fill in test_fills:
        mask = pygame.mask.Mask(size, fill=fill)
        for use_arg in (True, False):
            if use_arg:
                to_surface = mask.to_surface(None)
            else:
                to_surface = mask.to_surface()
            self.assertIsInstance(to_surface, pygame.Surface)
            if not IS_PYPY:
                self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
            self.assertTrue(to_surface.get_flags() & expected_flag)
            self.assertEqual(to_surface.get_bitsize(), expected_depth)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)