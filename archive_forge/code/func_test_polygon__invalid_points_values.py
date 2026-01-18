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
def test_polygon__invalid_points_values(self):
    """Ensures draw polygon handles invalid points values correctly."""
    kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'points': None, 'width': 0}
    points_fmts = (tuple(), ((1, 1),), ((1, 1), (2, 1)))
    for points in points_fmts:
        for seq_type in (tuple, list):
            kwargs['points'] = seq_type(points)
            with self.assertRaises(ValueError):
                bounds_rect = self.draw_polygon(**kwargs)