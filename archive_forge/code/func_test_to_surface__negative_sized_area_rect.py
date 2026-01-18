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
def test_to_surface__negative_sized_area_rect(self):
    """Ensures to_surface correctly handles negative sized area rects."""
    size = (3, 5)
    surface_color = pygame.Color('red')
    expected_color = pygame.Color('white')
    surface = pygame.Surface(size)
    mask = pygame.mask.Mask(size)
    mask.set_at((0, 0))
    areas = (pygame.Rect((0, 1), (1, -1)), pygame.Rect((1, 0), (-1, 1)), pygame.Rect((1, 1), (-1, -1)))
    for area in areas:
        surface.fill(surface_color)
        to_surface = mask.to_surface(surface, area=area)
        assertSurfaceFilled(self, to_surface, expected_color, area)
        assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, area)