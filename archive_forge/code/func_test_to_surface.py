from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
@unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
def test_to_surface(self):
    """Ensures empty and full masks can be drawn onto surfaces."""
    expected_ref_count = 3
    size = (33, 65)
    surface = pygame.Surface(size, SRCALPHA, 32)
    surface_color = pygame.Color('red')
    test_fills = ((pygame.Color('white'), True), (pygame.Color('black'), False))
    for expected_color, fill in test_fills:
        surface.fill(surface_color)
        mask = pygame.mask.Mask(size, fill=fill)
        to_surface = mask.to_surface(surface)
        self.assertIs(to_surface, surface)
        if not IS_PYPY:
            self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
        self.assertEqual(to_surface.get_size(), size)
        assertSurfaceFilled(self, to_surface, expected_color)