import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_project_v3_onto_other_as_list(self):
    """Project onto other list as vector."""
    v = Vector3(2, 3, 4)
    other = Vector3(3, 5, 7)
    actual = v.project(list(other))
    expected = v.dot(other) / other.dot(other) * other
    self.assertAlmostEqual(expected.x, actual.x)
    self.assertAlmostEqual(expected.y, actual.y)
    self.assertAlmostEqual(expected.z, actual.z)