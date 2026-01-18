import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_project_v3_onto_x_axis(self):
    """Project onto x-axis, e.g. get the component pointing in the x-axis direction."""
    v = Vector3(2, 3, 4)
    x_axis = Vector3(10, 0, 0)
    actual = v.project(x_axis)
    self.assertEqual(v.x, actual.x)
    self.assertEqual(0, actual.y)
    self.assertEqual(0, actual.z)