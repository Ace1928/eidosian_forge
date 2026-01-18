from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
@unittest.expectedFailure
@unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
def test_to_surface__area_off_mask(self):
    """Ensures area values off the mask work correctly
        when using the defaults for setcolor and unsetcolor.
        """
    default_setcolor = pygame.Color('white')
    default_unsetcolor = pygame.Color('black')
    width, height = size = (5, 7)
    surface = pygame.Surface(size, SRCALPHA, 32)
    surface_color = pygame.Color('red')
    positions = [(-width, -height), (-width, 0), (0, -height)]
    positions.extend(off_corners(pygame.Rect((0, 0), (width, height))))
    for fill in (True, False):
        mask = pygame.mask.Mask(size, fill=fill)
        mask_rect = mask.get_rect()
        area_rect = mask_rect.copy()
        expected_color = default_setcolor if fill else default_unsetcolor
        for pos in positions:
            surface.fill(surface_color)
            area_rect.topleft = pos
            overlap_rect = mask_rect.clip(area_rect)
            overlap_rect.topleft = (0, 0)
            to_surface = mask.to_surface(surface, area=area_rect)
            self.assertIs(to_surface, surface)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color, overlap_rect)
            assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, overlap_rect)