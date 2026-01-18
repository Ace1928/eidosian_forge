from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__depths(self):
    """Ensures to_surface works correctly with supported surface depths."""
    size = (13, 17)
    surface_color = pygame.Color('red')
    setsurface_color = pygame.Color('green')
    unsetsurface_color = pygame.Color('blue')
    for depth in (8, 16, 24, 32):
        surface = pygame.Surface(size, depth=depth)
        setsurface = pygame.Surface(size, depth=depth)
        unsetsurface = pygame.Surface(size, depth=depth)
        surface.fill(surface_color)
        setsurface.fill(setsurface_color)
        unsetsurface.fill(unsetsurface_color)
        for fill in (True, False):
            mask = pygame.mask.Mask(size, fill=fill)
            expected_color = setsurface.get_at((0, 0)) if fill else unsetsurface.get_at((0, 0))
            to_surface = mask.to_surface(surface, setsurface, unsetsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)