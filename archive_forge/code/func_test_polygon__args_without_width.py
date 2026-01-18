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
def test_polygon__args_without_width(self):
    """Ensures draw polygon accepts the args without a width."""
    bounds_rect = self.draw_polygon(pygame.Surface((2, 2)), (0, 0, 0, 50), ((0, 0), (1, 1), (2, 2)))
    self.assertIsInstance(bounds_rect, pygame.Rect)