from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__set_and_unset_bits_with_setsurface_unsetsurface(self):
    """Ensures that to_surface works correctly with with set/unset bits
        when using setsurface and unsetsurface.
        """
    width, height = size = (10, 20)
    mask = pygame.mask.Mask(size)
    mask_rect = mask.get_rect()
    surface = pygame.Surface(size)
    surface_color = pygame.Color('red')
    setsurface = surface.copy()
    setsurface_color = pygame.Color('green')
    setsurface.fill(setsurface_color)
    unsetsurface = surface.copy()
    unsetsurface_color = pygame.Color('blue')
    unsetsurface.fill(unsetsurface_color)
    for pos in ((x, y) for x in range(width) for y in range(x & 1, height, 2)):
        mask.set_at(pos)
    for dest in self.ORIGIN_OFFSETS:
        mask_rect.topleft = dest
        for disable_color_params in (True, False):
            surface.fill(surface_color)
            if disable_color_params:
                to_surface = mask.to_surface(surface, dest=dest, setsurface=setsurface, unsetsurface=unsetsurface, setcolor=None, unsetcolor=None)
            else:
                to_surface = mask.to_surface(surface, dest=dest, setsurface=setsurface, unsetsurface=unsetsurface)
            to_surface.lock()
            for pos in ((x, y) for x in range(width) for y in range(height)):
                mask_pos = (pos[0] - dest[0], pos[1] - dest[1])
                if not mask_rect.collidepoint(pos):
                    expected_color = surface_color
                elif mask.get_at(mask_pos):
                    expected_color = setsurface_color
                else:
                    expected_color = unsetsurface_color
                self.assertEqual(to_surface.get_at(pos), expected_color)
            to_surface.unlock()