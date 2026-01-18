import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_project_v2_onto_other(self):
    """Project onto other vector."""
    v = Vector2(2, 3)
    other = Vector2(3, 5)
    actual = v.project(other)
    expected = v.dot(other) / other.dot(other) * other
    self.assertEqual(expected.x, actual.x)
    self.assertEqual(expected.y, actual.y)