import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_swizzle_four_constants(self):
    """We can get 4 constant values from a swizzle."""
    v = Vector2(7, 6)
    self.assertEqual(v.xy01, (7.0, 6.0, 0.0, 1.0))