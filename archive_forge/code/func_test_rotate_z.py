import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_rotate_z(self):
    v1 = Vector3(1, 0, 0)
    v2 = v1.rotate_z(90)
    v3 = v1.rotate_z(90 + 360)
    self.assertEqual(v1.x, 1)
    self.assertEqual(v1.y, 0)
    self.assertEqual(v1.z, 0)
    self.assertAlmostEqual(v2.x, 0)
    self.assertAlmostEqual(v2.y, 1)
    self.assertEqual(v2.z, 0)
    self.assertAlmostEqual(v3.x, v2.x)
    self.assertAlmostEqual(v3.y, v2.y)
    self.assertEqual(v3.z, v2.z)
    v1 = Vector3(-1, -1, -1)
    v2 = v1.rotate_z(-90)
    self.assertAlmostEqual(v2.x, -1)
    self.assertAlmostEqual(v2.y, 1)
    self.assertEqual(v2.z, -1)
    v2 = v1.rotate_z(360)
    self.assertAlmostEqual(v1.x, v2.x)
    self.assertAlmostEqual(v1.y, v2.y)
    self.assertEqual(v1.z, v2.z)
    v2 = v1.rotate_z(0)
    self.assertAlmostEqual(v1.x, v2.x)
    self.assertAlmostEqual(v1.y, v2.y)
    self.assertEqual(v1.z, v2.z)