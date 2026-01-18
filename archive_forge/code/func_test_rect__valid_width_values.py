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
def test_rect__valid_width_values(self):
    """Ensures draw rect accepts different width values."""
    pos = (1, 1)
    surface_color = pygame.Color('black')
    surface = pygame.Surface((3, 4))
    color = (1, 2, 3, 255)
    kwargs = {'surface': surface, 'color': color, 'rect': pygame.Rect(pos, (2, 2)), 'width': None}
    for width in (-1000, -10, -1, 0, 1, 10, 1000):
        surface.fill(surface_color)
        kwargs['width'] = width
        expected_color = color if width >= 0 else surface_color
        bounds_rect = self.draw_rect(**kwargs)
        self.assertEqual(surface.get_at(pos), expected_color)
        self.assertIsInstance(bounds_rect, pygame.Rect)