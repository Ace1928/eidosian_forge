import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_scale_to_length(self):
    v = Vector3(1, 1, 1)
    v.scale_to_length(2.5)
    self.assertEqual(v, Vector3(2.5, 2.5, 2.5) / math.sqrt(3))
    self.assertRaises(ValueError, lambda: self.zeroVec.scale_to_length(1))
    self.assertEqual(v.scale_to_length(0), None)
    self.assertEqual(v, self.zeroVec)