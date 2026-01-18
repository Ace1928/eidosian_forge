from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__same_srcalphas_with_created_surfaces(self):
    """Ensures to_surface works correctly when it creates a surface
        and the SRCALPHA flag is set on both setsurface and unsetsurface.
        """
    size = (13, 17)
    setsurface_color = pygame.Color('green')
    unsetsurface_color = pygame.Color('blue')
    expected_flags = SRCALPHA
    setsurface = pygame.Surface(size, flags=expected_flags, depth=32)
    unsetsurface = pygame.Surface(size, flags=expected_flags, depth=32)
    setsurface.fill(setsurface_color)
    unsetsurface.fill(unsetsurface_color)
    for fill in (True, False):
        mask = pygame.mask.Mask(size, fill=fill)
        expected_color = setsurface_color if fill else unsetsurface_color
        to_surface = mask.to_surface(setsurface=setsurface, unsetsurface=unsetsurface)
        self.assertIsInstance(to_surface, pygame.Surface)
        self.assertEqual(to_surface.get_size(), size)
        assertSurfaceFilled(self, to_surface, expected_color)
        self.assertTrue(to_surface.get_flags() & expected_flags)