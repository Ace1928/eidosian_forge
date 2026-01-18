import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_unionall__invalid_rect_format(self):
    """Ensures unionall correctly handles invalid rect parameters."""
    numbers = [0, 1.2, 2, 3.3]
    strs = ['a', 'b', 'c']
    nones = [None, None]
    for invalid_rects in (numbers, strs, nones):
        with self.assertRaises(TypeError):
            Rect(0, 0, 1, 1).unionall(invalid_rects)