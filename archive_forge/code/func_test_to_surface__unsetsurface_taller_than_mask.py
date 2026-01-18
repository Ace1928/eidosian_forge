from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__unsetsurface_taller_than_mask(self):
    """Ensures that unsetsurfaces taller than the mask work correctly.

        For this test the unsetsurface's height is greater than the mask's
        height.
        """
    default_setcolor = pygame.Color('white')
    mask_size = (10, 6)
    tall_size = (10, 11)
    unsetsurface = pygame.Surface(tall_size, SRCALPHA, 32)
    unsetsurface_color = pygame.Color('red')
    unsetsurface.fill(unsetsurface_color)
    for fill in (True, False):
        mask = pygame.mask.Mask(mask_size, fill=fill)
        expected_color = default_setcolor if fill else unsetsurface_color
        to_surface = mask.to_surface(unsetsurface=unsetsurface)
        self.assertIsInstance(to_surface, pygame.Surface)
        self.assertEqual(to_surface.get_size(), mask_size)
        assertSurfaceFilled(self, to_surface, expected_color)