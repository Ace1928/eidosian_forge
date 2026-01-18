from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__surface_wider_than_mask(self):
    """Ensures that surfaces wider than the mask work correctly.

        For this test the surface's width is greater than the mask's width.
        """
    default_setcolor = pygame.Color('white')
    default_unsetcolor = pygame.Color('black')
    mask_size = (6, 15)
    wide_size = (11, 15)
    surface = pygame.Surface(wide_size)
    surface_color = pygame.Color('red')
    for fill in (True, False):
        mask = pygame.mask.Mask(mask_size, fill=fill)
        mask_rect = mask.get_rect()
        surface.fill(surface_color)
        expected_color = default_setcolor if fill else default_unsetcolor
        to_surface = mask.to_surface(surface)
        self.assertIs(to_surface, surface)
        self.assertEqual(to_surface.get_size(), wide_size)
        assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
        assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)