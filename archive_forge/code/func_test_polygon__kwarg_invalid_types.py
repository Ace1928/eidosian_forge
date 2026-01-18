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
def test_polygon__kwarg_invalid_types(self):
    """Ensures draw polygon detects invalid kwarg types."""
    surface = pygame.Surface((3, 3))
    color = pygame.Color('green')
    points = ((0, 0), (1, 0), (2, 0))
    width = 1
    kwargs_list = [{'surface': pygame.Surface, 'color': color, 'points': points, 'width': width}, {'surface': surface, 'color': 2.3, 'points': points, 'width': width}, {'surface': surface, 'color': color, 'points': ((1,), (1,), (1,)), 'width': width}, {'surface': surface, 'color': color, 'points': points, 'width': 1.2}]
    for kwargs in kwargs_list:
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon(**kwargs)