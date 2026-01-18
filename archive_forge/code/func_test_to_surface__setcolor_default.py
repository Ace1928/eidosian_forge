from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__setcolor_default(self):
    """Ensures the default setcolor is correct."""
    expected_color = pygame.Color('white')
    size = (3, 7)
    mask = pygame.mask.Mask(size, fill=True)
    to_surface = mask.to_surface(surface=None, setsurface=None, unsetsurface=None, unsetcolor=None)
    self.assertEqual(to_surface.get_size(), size)
    assertSurfaceFilled(self, to_surface, expected_color)