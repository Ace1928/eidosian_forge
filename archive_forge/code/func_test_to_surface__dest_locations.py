from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__dest_locations(self):
    """Ensures dest values can be different locations on/off the surface."""
    SIDE = 7
    surface = pygame.Surface((SIDE, SIDE))
    surface_rect = surface.get_rect()
    dest_rect = surface_rect.copy()
    surface_color = pygame.Color('red')
    default_setcolor = pygame.Color('white')
    default_unsetcolor = pygame.Color('black')
    directions = (((s, 0) for s in range(-SIDE, SIDE + 1)), ((0, s) for s in range(-SIDE, SIDE + 1)), ((s, s) for s in range(-SIDE, SIDE + 1)), ((-s, s) for s in range(-SIDE, SIDE + 1)))
    for fill in (True, False):
        mask = pygame.mask.Mask((SIDE, SIDE), fill=fill)
        expected_color = default_setcolor if fill else default_unsetcolor
        for direction in directions:
            for pos in direction:
                dest_rect.topleft = pos
                overlap_rect = dest_rect.clip(surface_rect)
                surface.fill(surface_color)
                to_surface = mask.to_surface(surface, dest=dest_rect)
                assertSurfaceFilled(self, to_surface, expected_color, overlap_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, overlap_rect)