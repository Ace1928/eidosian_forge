from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__surface_color_alphas(self):
    """Ensures the setsurface/unsetsurface color alpha values are respected."""
    size = (13, 17)
    setsurface_color = pygame.Color('green')
    setsurface_color.a = 53
    unsetsurface_color = pygame.Color('blue')
    unsetsurface_color.a = 109
    setsurface = pygame.Surface(size, flags=SRCALPHA, depth=32)
    unsetsurface = pygame.Surface(size, flags=SRCALPHA, depth=32)
    setsurface.fill(setsurface_color)
    unsetsurface.fill(unsetsurface_color)
    for fill in (True, False):
        mask = pygame.mask.Mask(size, fill=fill)
        expected_color = setsurface_color if fill else unsetsurface_color
        to_surface = mask.to_surface(setsurface=setsurface, unsetsurface=unsetsurface)
        self.assertIsInstance(to_surface, pygame.Surface)
        self.assertEqual(to_surface.get_size(), size)
        assertSurfaceFilled(self, to_surface, expected_color)