from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__set_and_unset_bits(self):
    """Ensures that to_surface works correctly with with set/unset bits
        when using the defaults for setcolor and unsetcolor.
        """
    default_setcolor = pygame.Color('white')
    default_unsetcolor = pygame.Color('black')
    width, height = size = (10, 20)
    mask = pygame.mask.Mask(size)
    mask_rect = mask.get_rect()
    surface = pygame.Surface(size)
    surface_color = pygame.Color('red')
    for pos in ((x, y) for x in range(width) for y in range(x & 1, height, 2)):
        mask.set_at(pos)
    for dest in self.ORIGIN_OFFSETS:
        mask_rect.topleft = dest
        surface.fill(surface_color)
        to_surface = mask.to_surface(surface, dest=dest)
        to_surface.lock()
        for pos in ((x, y) for x in range(width) for y in range(height)):
            mask_pos = (pos[0] - dest[0], pos[1] - dest[1])
            if not mask_rect.collidepoint(pos):
                expected_color = surface_color
            elif mask.get_at(mask_pos):
                expected_color = default_setcolor
            else:
                expected_color = default_unsetcolor
            self.assertEqual(to_surface.get_at(pos), expected_color, (dest, pos))
        to_surface.unlock()