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
def test_rect__kwargs_order_independent(self):
    """Ensures draw rect's kwargs are not order dependent."""
    bounds_rect = self.draw_rect(color=(0, 1, 2), border_radius=10, surface=pygame.Surface((2, 3)), border_top_left_radius=5, width=-2, border_top_right_radius=20, border_bottom_right_radius=0, rect=pygame.Rect((0, 0), (0, 0)), border_bottom_left_radius=15)
    self.assertIsInstance(bounds_rect, pygame.Rect)