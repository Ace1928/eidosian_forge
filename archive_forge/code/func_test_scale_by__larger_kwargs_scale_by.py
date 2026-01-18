import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_scale_by__larger_kwargs_scale_by(self):
    """
        The scale method scales around the center of the rectangle
        Uses 'scale_by' kwarg.
        """
    r = Rect(2, 4, 6, 8)
    r2 = r.scale_by(scale_by=(2, 4))
    self.assertEqual(r.center, r2.center)
    self.assertEqual(r.left - 3, r2.left)
    self.assertEqual(r.centery - r.h * 4 / 2, r2.top)
    self.assertEqual(r.right + 3, r2.right)
    self.assertEqual(r.centery + r.h * 4 / 2, r2.bottom)
    self.assertEqual(r.width * 2, r2.width)
    self.assertEqual(r.height * 4, r2.height)