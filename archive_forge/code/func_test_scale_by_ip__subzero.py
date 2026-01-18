import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_scale_by_ip__subzero(self):
    """The scale method scales around the center of the rectangle"""
    r = Rect(2, 4, 6, 8)
    r.scale_by_ip(0)
    r.scale_by_ip(-1)
    r.scale_by_ip(-1e-06)
    r.scale_by_ip(1e-05)