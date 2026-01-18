from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
@unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
def test_to_surface__all_surfaces_different_sizes_than_mask(self):
    """Ensures that all the surface parameters can be of different sizes."""
    default_setcolor = pygame.Color('white')
    default_unsetcolor = pygame.Color('black')
    surface_color = pygame.Color('red')
    setsurface_color = pygame.Color('green')
    unsetsurface_color = pygame.Color('blue')
    mask_size = (10, 15)
    surface_size = (11, 14)
    setsurface_size = (9, 8)
    unsetsurface_size = (12, 16)
    surface = pygame.Surface(surface_size)
    setsurface = pygame.Surface(setsurface_size)
    unsetsurface = pygame.Surface(unsetsurface_size)
    surface.fill(surface_color)
    setsurface.fill(setsurface_color)
    unsetsurface.fill(unsetsurface_color)
    surface_rect = surface.get_rect()
    setsurface_rect = setsurface.get_rect()
    unsetsurface_rect = unsetsurface.get_rect()
    mask = pygame.mask.Mask(mask_size, fill=True)
    mask_rect = mask.get_rect()
    unfilled_rect = pygame.Rect((0, 0), (4, 5))
    unfilled_rect.center = mask_rect.center
    for pos in ((x, y) for x in range(unfilled_rect.x, unfilled_rect.w) for y in range(unfilled_rect.y, unfilled_rect.h)):
        mask.set_at(pos, 0)
    to_surface = mask.to_surface(surface, setsurface, unsetsurface)
    self.assertIs(to_surface, surface)
    self.assertEqual(to_surface.get_size(), surface_size)
    to_surface.lock()
    for pos in ((x, y) for x in range(surface_rect.w) for y in range(surface_rect.h)):
        if not mask_rect.collidepoint(pos):
            expected_color = surface_color
        elif mask.get_at(pos):
            if setsurface_rect.collidepoint(pos):
                expected_color = setsurface_color
            else:
                expected_color = default_setcolor
        elif unsetsurface_rect.collidepoint(pos):
            expected_color = unsetsurface_color
        else:
            expected_color = default_unsetcolor
        self.assertEqual(to_surface.get_at(pos), expected_color)
    to_surface.unlock()