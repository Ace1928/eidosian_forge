import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_y(self):
    """Ensures changing the y attribute moves the rect and does not change
        the rect's size.
        """
    expected_x = 1
    expected_y = 20
    expected_size = (3, 4)
    r = Rect((expected_x, 2), expected_size)
    r.y = expected_y
    self.assertEqual(r.y, expected_y)
    self.assertEqual(r.y, r.top)
    self.assertEqual(r.x, expected_x)
    self.assertEqual(r.size, expected_size)