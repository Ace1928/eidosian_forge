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
def test_rect__args_and_kwargs(self):
    """Ensures draw rect accepts a combination of args/kwargs"""
    surface = pygame.Surface((3, 1))
    color = (255, 255, 255, 0)
    rect = pygame.Rect((1, 0), (2, 5))
    width = 0
    kwargs = {'surface': surface, 'color': color, 'rect': rect, 'width': width}
    for name in ('surface', 'color', 'rect', 'width'):
        kwargs.pop(name)
        if 'surface' == name:
            bounds_rect = self.draw_rect(surface, **kwargs)
        elif 'color' == name:
            bounds_rect = self.draw_rect(surface, color, **kwargs)
        elif 'rect' == name:
            bounds_rect = self.draw_rect(surface, color, rect, **kwargs)
        else:
            bounds_rect = self.draw_rect(surface, color, rect, width, **kwargs)
        self.assertIsInstance(bounds_rect, pygame.Rect)