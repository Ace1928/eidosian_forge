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
def test_rect__kwargs(self):
    """Ensures draw rect accepts the correct kwargs
        with and without a width and border_radius arg.
        """
    kwargs_list = [{'surface': pygame.Surface((5, 5)), 'color': pygame.Color('red'), 'rect': pygame.Rect((0, 0), (1, 2)), 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': pygame.Surface((1, 2)), 'color': (0, 100, 200), 'rect': (0, 0, 1, 1)}]
    for kwargs in kwargs_list:
        bounds_rect = self.draw_rect(**kwargs)
        self.assertIsInstance(bounds_rect, pygame.Rect)