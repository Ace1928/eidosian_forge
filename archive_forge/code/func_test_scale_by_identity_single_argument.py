import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_scale_by_identity_single_argument(self):
    """The scale method scales around the center of the rectangle"""
    r = Rect(2, 4, 6, 8)
    actual = r.scale_by(1)
    self.assertEqual(r.x, actual.x)
    self.assertEqual(r.y, actual.y)
    self.assertEqual(r.w, actual.w)
    self.assertEqual(r.h, actual.h)