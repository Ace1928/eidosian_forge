from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__surface_shorter_than_mask(self):
    """Ensures that surfaces shorter than the mask work correctly.

        For this test the surface's height is less than the mask's height.
        """
    default_setcolor = pygame.Color('white')
    default_unsetcolor = pygame.Color('black')
    mask_size = (10, 11)
    short_size = (10, 6)
    surface = pygame.Surface(short_size)
    surface_color = pygame.Color('red')
    for fill in (True, False):
        mask = pygame.mask.Mask(mask_size, fill=fill)
        surface.fill(surface_color)
        expected_color = default_setcolor if fill else default_unsetcolor
        to_surface = mask.to_surface(surface)
        self.assertIs(to_surface, surface)
        self.assertEqual(to_surface.get_size(), short_size)
        assertSurfaceFilled(self, to_surface, expected_color)