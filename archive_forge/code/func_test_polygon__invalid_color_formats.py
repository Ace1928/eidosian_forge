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
def test_polygon__invalid_color_formats(self):
    """Ensures draw polygon handles invalid color formats correctly."""
    kwargs = {'surface': pygame.Surface((4, 3)), 'color': None, 'points': ((1, 1), (2, 1), (2, 2), (1, 2)), 'width': 0}
    for expected_color in (2.3, self):
        kwargs['color'] = expected_color
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon(**kwargs)