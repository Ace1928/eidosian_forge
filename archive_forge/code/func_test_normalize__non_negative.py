import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_normalize__non_negative(self):
    """Ensures normalize works when width and height are both non-negative.

        Tests combinations of positive and zero values for width and height.
        The normalize method has no impact when both width and height are
        non-negative.
        """
    for size in ((3, 6), (3, 0), (0, 6), (0, 0)):
        test_rect = Rect((1, 2), size)
        expected_normalized_rect = Rect(test_rect)
        test_rect.normalize()
        self.assertEqual(test_rect, expected_normalized_rect)