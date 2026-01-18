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
def test_to_surface__area_on_mask_with_setsurface_unsetsurface(self):
    """Ensures area values on the mask work correctly
        when using setsurface and unsetsurface.
        """
    width, height = size = (5, 9)
    surface = pygame.Surface(size, SRCALPHA, 32)
    surface_color = pygame.Color('red')
    setsurface = surface.copy()
    setsurface_color = pygame.Color('green')
    setsurface.fill(setsurface_color)
    unsetsurface = surface.copy()
    unsetsurface_color = pygame.Color('blue')
    unsetsurface.fill(unsetsurface_color)
    kwargs = {'surface': surface, 'setsurface': setsurface, 'unsetsurface': unsetsurface, 'area': pygame.Rect((0, 0), size)}
    color_kwargs = dict(kwargs)
    color_kwargs.update((('setcolor', None), ('unsetcolor', None)))
    for fill in (True, False):
        mask = pygame.mask.Mask(size, fill=fill)
        mask_rect = mask.get_rect()
        area_rect = mask_rect.copy()
        expected_color = setsurface_color if fill else unsetsurface_color
        for pos in ((x, y) for y in range(height) for x in range(width)):
            area_rect.topleft = pos
            overlap_rect = mask_rect.clip(area_rect)
            overlap_rect.topleft = (0, 0)
            for use_color_params in (True, False):
                surface.fill(surface_color)
                test_kwargs = color_kwargs if use_color_params else kwargs
                test_kwargs['area'].topleft = pos
                overlap_rect = mask_rect.clip(test_kwargs['area'])
                overlap_rect.topleft = (0, 0)
                to_surface = mask.to_surface(**test_kwargs)
                self.assertIs(to_surface, surface)
                self.assertEqual(to_surface.get_size(), size)
                assertSurfaceFilled(self, to_surface, expected_color, overlap_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, overlap_rect)