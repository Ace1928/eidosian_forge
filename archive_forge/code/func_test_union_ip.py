import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_union_ip(self):
    r1 = Rect(1, 1, 1, 2)
    r2 = Rect(-2, -2, 1, 2)
    r1.union_ip(r2)
    self.assertEqual(Rect(-2, -2, 4, 5), r1)