from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__kwargs_order_independent(self):
    """Ensures to_surface kwargs are not order dependent."""
    expected_color = pygame.Color('blue')
    size = (3, 2)
    mask = pygame.mask.Mask(size, fill=True)
    surface = pygame.Surface(size)
    to_surface = mask.to_surface(dest=(0, 0), setcolor=expected_color, unsetcolor=None, surface=surface, unsetsurface=pygame.Surface(size), setsurface=None)
    self.assertIs(to_surface, surface)
    self.assertEqual(to_surface.get_size(), size)
    assertSurfaceFilled(self, to_surface, expected_color)