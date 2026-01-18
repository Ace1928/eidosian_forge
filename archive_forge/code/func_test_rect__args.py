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
def test_rect__args(self):
    """Ensures draw rect accepts the correct args."""
    bounds_rect = self.draw_rect(pygame.Surface((2, 2)), (20, 10, 20, 150), pygame.Rect((0, 0), (1, 1)), 2, 1, 2, 3, 4, 5)
    self.assertIsInstance(bounds_rect, pygame.Rect)