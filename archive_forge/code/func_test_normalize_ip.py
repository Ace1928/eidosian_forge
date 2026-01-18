import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_normalize_ip(self):
    v = +self.v1
    self.assertNotEqual(v.x * v.x + v.y * v.y + v.z * v.z, 1.0)
    self.assertEqual(v.normalize_ip(), None)
    self.assertAlmostEqual(v.x * v.x + v.y * v.y + v.z * v.z, 1.0)
    cross = (self.v1.y * v.z - self.v1.z * v.y) ** 2 + (self.v1.z * v.x - self.v1.x * v.z) ** 2 + (self.v1.x * v.y - self.v1.y * v.x) ** 2
    self.assertAlmostEqual(cross, 0.0)
    self.assertRaises(ValueError, lambda: self.zeroVec.normalize_ip())