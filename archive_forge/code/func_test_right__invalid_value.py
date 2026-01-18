import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_right__invalid_value(self):
    """Ensures the right attribute handles invalid values correctly."""
    r = Rect(0, 0, 1, 1)
    for value in (None, [], '1', (1,), [1, 2, 3]):
        with self.assertRaises(TypeError):
            r.right = value