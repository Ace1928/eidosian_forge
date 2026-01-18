import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_rotate_x_ip(self):
    v = Vector3(1, 0, 0)
    self.assertEqual(v.rotate_x_ip(90), None)
    self.assertEqual(v.x, 1)
    self.assertEqual(v.y, 0)
    self.assertEqual(v.z, 0)
    v = Vector3(-1, -1, 1)
    v.rotate_x_ip(-90)
    self.assertEqual(v.x, -1)
    self.assertAlmostEqual(v.y, 1)
    self.assertAlmostEqual(v.z, 1)