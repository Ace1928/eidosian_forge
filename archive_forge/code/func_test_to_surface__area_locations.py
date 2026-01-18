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
def test_to_surface__area_locations(self):
    """Ensures area rects can be different locations on/off the mask."""
    SIDE = 7
    surface = pygame.Surface((SIDE, SIDE))
    surface_color = pygame.Color('red')
    default_setcolor = pygame.Color('white')
    default_unsetcolor = pygame.Color('black')
    directions = (((s, 0) for s in range(-SIDE, SIDE + 1)), ((0, s) for s in range(-SIDE, SIDE + 1)), ((s, s) for s in range(-SIDE, SIDE + 1)), ((-s, s) for s in range(-SIDE, SIDE + 1)))
    for fill in (True, False):
        mask = pygame.mask.Mask((SIDE, SIDE), fill=fill)
        mask_rect = mask.get_rect()
        area_rect = mask_rect.copy()
        expected_color = default_setcolor if fill else default_unsetcolor
        for direction in directions:
            for pos in direction:
                area_rect.topleft = pos
                overlap_rect = area_rect.clip(mask_rect)
                overlap_rect.topleft = (0, 0)
                surface.fill(surface_color)
                to_surface = mask.to_surface(surface, area=area_rect)
                assertSurfaceFilled(self, to_surface, expected_color, overlap_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, overlap_rect)