import math
import unittest
import sys
import warnings
import pygame
from pygame import draw
from pygame import draw_py
from pygame.locals import SRCALPHA
from pygame.tests import test_utils
from pygame.math import Vector2
def test_polygon__arg_invalid_types(self):
    """Ensures draw polygon detects invalid arg types."""
    surface = pygame.Surface((2, 2))
    color = pygame.Color('blue')
    points = ((0, 1), (1, 2), (1, 3))
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_polygon(surface, color, points, '1')
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_polygon(surface, color, (1, 2, 3))
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_polygon(surface, 2.3, points)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_polygon((1, 2, 3, 4), color, points)