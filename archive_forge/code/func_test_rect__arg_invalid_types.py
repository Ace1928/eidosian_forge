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
def test_rect__arg_invalid_types(self):
    """Ensures draw rect detects invalid arg types."""
    surface = pygame.Surface((3, 3))
    color = pygame.Color('white')
    rect = pygame.Rect((1, 1), (1, 1))
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_rect(surface, color, rect, 2, border_bottom_right_radius='rad')
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_rect(surface, color, rect, 2, border_bottom_left_radius='rad')
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_rect(surface, color, rect, 2, border_top_right_radius='rad')
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_rect(surface, color, rect, 2, border_top_left_radius='draw')
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_rect(surface, color, rect, 2, 'rad')
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_rect(surface, color, rect, '2', 4)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_rect(surface, color, (1, 2, 3), 2, 6)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_rect(surface, 2.3, rect, 3, 8)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_rect(rect, color, rect, 4, 10)