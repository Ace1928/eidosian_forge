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
def test_rect__one_pixel_lines(self):
    self.surf = pygame.Surface((320, 200), pygame.SRCALPHA)
    self.color = (1, 13, 24, 205)
    rect = pygame.Rect(10, 10, 56, 20)
    drawn = self.draw_rect(self.surf, self.color, rect, 1)
    self.assertEqual(drawn, rect)
    for pt in test_utils.rect_perimeter_pts(drawn):
        color_at_pt = self.surf.get_at(pt)
        self.assertEqual(color_at_pt, self.color)
    for pt in test_utils.rect_outer_bounds(drawn):
        color_at_pt = self.surf.get_at(pt)
        self.assertNotEqual(color_at_pt, self.color)