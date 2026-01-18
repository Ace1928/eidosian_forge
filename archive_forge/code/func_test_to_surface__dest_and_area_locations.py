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
def test_to_surface__dest_and_area_locations(self):
    """Ensures dest/area values can be different locations on/off the
        surface/mask.
        """
    SIDE = 5
    surface = pygame.Surface((SIDE, SIDE))
    surface_rect = surface.get_rect()
    dest_rect = surface_rect.copy()
    surface_color = pygame.Color('red')
    default_setcolor = pygame.Color('white')
    default_unsetcolor = pygame.Color('black')
    dest_directions = (((s, 0) for s in range(-SIDE, SIDE + 1)), ((0, s) for s in range(-SIDE, SIDE + 1)), ((s, s) for s in range(-SIDE, SIDE + 1)), ((-s, s) for s in range(-SIDE, SIDE + 1)))
    area_positions = list(dest_directions[2])
    for fill in (True, False):
        mask = pygame.mask.Mask((SIDE, SIDE), fill=fill)
        mask_rect = mask.get_rect()
        area_rect = mask_rect.copy()
        expected_color = default_setcolor if fill else default_unsetcolor
        for dest_direction in dest_directions:
            for dest_pos in dest_direction:
                dest_rect.topleft = dest_pos
                for area_pos in area_positions:
                    area_rect.topleft = area_pos
                    area_overlap_rect = area_rect.clip(mask_rect)
                    area_overlap_rect.topleft = dest_rect.topleft
                    dest_overlap_rect = dest_rect.clip(area_overlap_rect)
                    surface.fill(surface_color)
                    to_surface = mask.to_surface(surface, dest=dest_rect, area=area_rect)
                    assertSurfaceFilled(self, to_surface, expected_color, dest_overlap_rect)
                    assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, dest_overlap_rect)