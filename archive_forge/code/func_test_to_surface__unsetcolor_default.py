from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__unsetcolor_default(self):
    """Ensures the default unsetcolor is correct."""
    expected_color = pygame.Color('black')
    size = (3, 7)
    mask = pygame.mask.Mask(size)
    to_surface = mask.to_surface(surface=None, setsurface=None, unsetsurface=None, setcolor=None)
    self.assertEqual(to_surface.get_size(), size)
    assertSurfaceFilled(self, to_surface, expected_color)