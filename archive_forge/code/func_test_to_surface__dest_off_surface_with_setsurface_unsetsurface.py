from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__dest_off_surface_with_setsurface_unsetsurface(self):
    """Ensures dest values off the surface work correctly
        when using setsurface and unsetsurface.
        """
    width, height = size = (5, 7)
    surface = pygame.Surface(size, SRCALPHA, 32)
    surface_color = pygame.Color('red')
    setsurface = surface.copy()
    setsurface_color = pygame.Color('green')
    setsurface.fill(setsurface_color)
    unsetsurface = surface.copy()
    unsetsurface_color = pygame.Color('blue')
    unsetsurface.fill(unsetsurface_color)
    dests = [(-width, -height), (-width, 0), (0, -height)]
    dests.extend(off_corners(surface.get_rect()))
    kwargs = {'surface': surface, 'setsurface': setsurface, 'unsetsurface': unsetsurface, 'dest': None}
    color_kwargs = dict(kwargs)
    color_kwargs.update((('setcolor', None), ('unsetcolor', None)))
    for fill in (True, False):
        mask = pygame.mask.Mask(size, fill=fill)
        mask_rect = mask.get_rect()
        expected_color = setsurface_color if fill else unsetsurface_color
        for dest in dests:
            mask_rect.topleft = dest
            for use_color_params in (True, False):
                surface.fill(surface_color)
                test_kwargs = color_kwargs if use_color_params else kwargs
                test_kwargs['dest'] = dest
                to_surface = mask.to_surface(**test_kwargs)
                self.assertIs(to_surface, surface)
                self.assertEqual(to_surface.get_size(), size)
                assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)