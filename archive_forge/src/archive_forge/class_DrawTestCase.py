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
class DrawTestCase(unittest.TestCase):
    """Base class to test draw module functions."""
    draw_rect = staticmethod(draw.rect)
    draw_polygon = staticmethod(draw.polygon)
    draw_circle = staticmethod(draw.circle)
    draw_ellipse = staticmethod(draw.ellipse)
    draw_arc = staticmethod(draw.arc)
    draw_line = staticmethod(draw.line)
    draw_lines = staticmethod(draw.lines)
    draw_aaline = staticmethod(draw.aaline)
    draw_aalines = staticmethod(draw.aalines)