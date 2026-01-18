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
def test_polygon__kwargs_order_independent(self):
    """Ensures draw polygon's kwargs are not order dependent."""
    bounds_rect = self.draw_polygon(color=(10, 20, 30), surface=pygame.Surface((3, 2)), width=0, points=((0, 1), (1, 2), (2, 3)))
    self.assertIsInstance(bounds_rect, pygame.Rect)