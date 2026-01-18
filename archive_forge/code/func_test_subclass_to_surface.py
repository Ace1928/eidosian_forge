from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_to_surface(self):
    """Ensures to_surface works for subclassed Masks."""
    expected_color = pygame.Color('blue')
    size = (5, 3)
    mask = SubMask(size, fill=True)
    surface = pygame.Surface(size, SRCALPHA, 32)
    surface.fill(pygame.Color('red'))
    to_surface = mask.to_surface(surface, setcolor=expected_color)
    self.assertIs(to_surface, surface)
    self.assertEqual(to_surface.get_size(), size)
    assertSurfaceFilled(self, to_surface, expected_color)