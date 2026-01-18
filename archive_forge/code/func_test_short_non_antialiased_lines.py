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
def test_short_non_antialiased_lines(self):
    """test very short not anti aliased lines in all directions."""
    self.surface = pygame.Surface((10, 10))
    draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)
    check_points = [(i, j) for i in range(3, 8) for j in range(3, 8)]

    def check_both_directions(from_pt, to_pt, other_points):
        should = {pt: FG_GREEN for pt in other_points}
        self._check_antialiasing(from_pt, to_pt, should, check_points)
    check_both_directions((5, 5), (5, 5), [])
    check_both_directions((4, 7), (5, 7), [])
    check_both_directions((5, 4), (7, 4), [(6, 4)])
    check_both_directions((5, 5), (5, 6), [])
    check_both_directions((6, 4), (6, 6), [(6, 5)])
    check_both_directions((5, 5), (6, 6), [])
    check_both_directions((5, 5), (7, 7), [(6, 6)])
    check_both_directions((5, 6), (6, 5), [])
    check_both_directions((6, 4), (4, 6), [(5, 5)])