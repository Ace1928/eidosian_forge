import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_rotate_x_rad_ip(self):
    vec = Vector3(0, 1, 0)
    vec.rotate_x_rad_ip(math.pi / 2)
    self.assertEqual(vec, (0, 0, 1))