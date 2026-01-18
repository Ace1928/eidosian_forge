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
def test_polygon__args_missing(self):
    """Ensures draw polygon detects any missing required args."""
    surface = pygame.Surface((1, 1))
    color = pygame.Color('blue')
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_polygon(surface, color)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_polygon(surface)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_polygon()