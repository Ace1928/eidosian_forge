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
def test_rect__draw_line_width(self):
    surface = pygame.Surface((100, 100))
    surface.fill('black')
    color = pygame.Color(255, 255, 255)
    rect_width = 80
    rect_height = 50
    line_width = 10
    pygame.draw.rect(surface, color, pygame.Rect(0, 0, rect_width, rect_height), line_width)
    for i in range(line_width):
        self.assertEqual(surface.get_at((i, i)), color)
        self.assertEqual(surface.get_at((rect_width - i - 1, i)), color)
        self.assertEqual(surface.get_at((i, rect_height - i - 1)), color)
        self.assertEqual(surface.get_at((rect_width - i - 1, rect_height - i - 1)), color)
    self.assertEqual(surface.get_at((line_width, line_width)), (0, 0, 0))
    self.assertEqual(surface.get_at((rect_width - line_width - 1, line_width)), (0, 0, 0))
    self.assertEqual(surface.get_at((line_width, rect_height - line_width - 1)), (0, 0, 0))
    self.assertEqual(surface.get_at((rect_width - line_width - 1, rect_height - line_width - 1)), (0, 0, 0))