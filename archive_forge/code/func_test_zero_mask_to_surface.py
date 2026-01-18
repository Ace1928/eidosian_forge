from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_to_surface(self):
    """Ensures to_surface correctly handles zero sized masks and surfaces."""
    mask_color = pygame.Color('blue')
    surf_color = pygame.Color('red')
    for surf_size in ((7, 3), (7, 0), (0, 7), (0, 0)):
        surface = pygame.Surface(surf_size, SRCALPHA, 32)
        surface.fill(surf_color)
        for mask_size in ((5, 0), (0, 5), (0, 0)):
            mask = pygame.mask.Mask(mask_size, fill=True)
            to_surface = mask.to_surface(surface, setcolor=mask_color)
            self.assertIs(to_surface, surface)
            self.assertEqual(to_surface.get_size(), surf_size)
            if 0 not in surf_size:
                assertSurfaceFilled(self, to_surface, surf_color)