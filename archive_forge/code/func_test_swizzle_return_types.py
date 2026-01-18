import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_swizzle_return_types(self):
    self.assertEqual(type(self.v1.x), float)
    self.assertEqual(type(self.v1.xy), Vector2)
    self.assertEqual(type(self.v1.xyz), Vector3)
    self.assertEqual(type(self.v1.xyxy), tuple)
    self.assertEqual(type(self.v1.xyxyx), tuple)