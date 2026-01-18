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
def test_rect__invalid_rect_formats(self):
    """Ensures draw rect handles invalid rect formats correctly."""
    kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'rect': None, 'width': 0}
    invalid_fmts = ([], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4, 5], {1, 2, 3, 4}, [1, 2, 3, '4'])
    for rect in invalid_fmts:
        kwargs['rect'] = rect
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(**kwargs)