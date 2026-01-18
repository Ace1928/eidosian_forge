import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_scale_by_ip__kwarg_exceptions(self):
    """The scale method scales around the center of the rectangle using
        keyword argument 'scale_by'. Tests for incorrect keyword args"""
    r = Rect(2, 4, 6, 8)
    with self.assertRaises(TypeError):
        r.scale_by_ip(scale_by=2)
    with self.assertRaises(TypeError):
        r.scale_by_ip(scale_by=(1, 2), y=1)