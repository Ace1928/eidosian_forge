import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_update__TopLeftSize(self):
    """Test update with 2 tuples((x, y), (w, h))"""
    rect = Rect(0, 0, 1, 1)
    rect.update((1, 2), (3, 4))
    self.assertEqual(1, rect.left)
    self.assertEqual(2, rect.top)
    self.assertEqual(3, rect.width)
    self.assertEqual(4, rect.height)