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
def test_path_data_validation(self):
    """Test validation of multi-point drawing methods.

        See bug #521
        """
    surf = pygame.Surface((5, 5))
    rect = pygame.Rect(0, 0, 5, 5)
    bad_values = ('text', b'bytes', 1 + 1j, object(), lambda x: x)
    bad_points = list(bad_values) + [(1,), (1, 2, 3)]
    bad_points.extend(((1, v) for v in bad_values))
    good_path = [(1, 1), (1, 3), (3, 3), (3, 1)]
    check_pts = [(x, y) for x in range(5) for y in range(5)]
    for method, is_polgon in ((draw.lines, 0), (draw.aalines, 0), (draw.polygon, 1)):
        for val in bad_values:
            draw.rect(surf, RED, rect, 0)
            with self.assertRaises(TypeError):
                if is_polgon:
                    method(surf, GREEN, [val] + good_path, 0)
                else:
                    method(surf, GREEN, True, [val] + good_path)
            self.assertTrue(all((surf.get_at(pt) == RED for pt in check_pts)))
            draw.rect(surf, RED, rect, 0)
            with self.assertRaises(TypeError):
                path = good_path[:2] + [val] + good_path[2:]
                if is_polgon:
                    method(surf, GREEN, path, 0)
                else:
                    method(surf, GREEN, True, path)
            self.assertTrue(all((surf.get_at(pt) == RED for pt in check_pts)))