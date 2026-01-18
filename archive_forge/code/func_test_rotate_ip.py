import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_rotate_ip(self):
    v = Vector3(1, 0, 0)
    axis = Vector3(0, 1, 0)
    self.assertEqual(v.rotate_ip(90, axis), None)
    self.assertEqual(v.x, 0)
    self.assertEqual(v.y, 0)
    self.assertEqual(v.z, -1)
    v = Vector3(-1, -1, 1)
    v.rotate_ip(-90, axis)
    self.assertEqual(v.x, -1)
    self.assertEqual(v.y, -1)
    self.assertEqual(v.z, -1)