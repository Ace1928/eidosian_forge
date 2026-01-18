import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_subclassing_v3(self):
    """Check if Vector3 is subclassable"""
    v = Vector3(4, 2, 0)

    class TestVector(Vector3):

        def supermariobrosiscool(self):
            return 722
    other = TestVector(4, 1, 0)
    self.assertEqual(other.supermariobrosiscool(), 722)
    self.assertNotEqual(type(v), TestVector)
    self.assertNotEqual(type(v), type(other.copy()))
    self.assertEqual(TestVector, type(other.reflect(v)))
    self.assertEqual(TestVector, type(other.lerp(v, 1)))
    self.assertEqual(TestVector, type(other.slerp(v, 1)))
    self.assertEqual(TestVector, type(other.rotate(5, v)))
    self.assertEqual(TestVector, type(other.rotate_rad(5, v)))
    self.assertEqual(TestVector, type(other.project(v)))
    self.assertEqual(TestVector, type(other.move_towards(v, 5)))
    self.assertEqual(TestVector, type(other.clamp_magnitude(5)))
    self.assertEqual(TestVector, type(other.clamp_magnitude(1, 5)))
    self.assertEqual(TestVector, type(other.elementwise() + other))
    other1 = TestVector(4, 2, 0)
    self.assertEqual(type(other + other1), TestVector)
    self.assertEqual(type(other - other1), TestVector)
    self.assertEqual(type(other * 3), TestVector)
    self.assertEqual(type(other / 3), TestVector)
    self.assertEqual(type(other.elementwise() ** 3), TestVector)