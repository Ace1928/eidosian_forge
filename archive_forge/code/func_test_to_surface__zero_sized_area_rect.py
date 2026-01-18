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
def test_to_surface__zero_sized_area_rect(self):
    """Ensures to_surface correctly handles zero sized area rects."""
    size = (3, 5)
    expected_color = pygame.Color('red')
    surface = pygame.Surface(size)
    mask = pygame.mask.Mask(size, fill=True)
    areas = (pygame.Rect((0, 0), (0, 1)), pygame.Rect((0, 0), (1, 0)), pygame.Rect((0, 0), (0, 0)))
    for area in areas:
        surface.fill(expected_color)
        to_surface = mask.to_surface(surface, area=area)
        assertSurfaceFilled(self, to_surface, expected_color)