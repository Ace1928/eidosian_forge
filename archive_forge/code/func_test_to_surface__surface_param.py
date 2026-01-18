from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__surface_param(self):
    """Ensures to_surface accepts a surface arg/kwarg."""
    expected_ref_count = 4
    expected_color = pygame.Color('white')
    surface_color = pygame.Color('red')
    size = (5, 3)
    mask = pygame.mask.Mask(size, fill=True)
    surface = pygame.Surface(size)
    kwargs = {'surface': surface}
    for use_kwargs in (True, False):
        surface.fill(surface_color)
        if use_kwargs:
            to_surface = mask.to_surface(**kwargs)
        else:
            to_surface = mask.to_surface(kwargs['surface'])
        self.assertIs(to_surface, surface)
        if not IS_PYPY:
            self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
        self.assertEqual(to_surface.get_size(), size)
        assertSurfaceFilled(self, to_surface, expected_color)