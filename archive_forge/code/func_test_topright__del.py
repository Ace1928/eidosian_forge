import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_topright__del(self):
    """Ensures the topright attribute can't be deleted."""
    r = Rect(0, 0, 1, 1)
    with self.assertRaises(AttributeError):
        del r.topright