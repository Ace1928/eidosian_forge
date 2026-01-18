import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_project_v2_onto_y_axis(self):
    """Project onto y-axis, e.g. get the component pointing in the y-axis direction."""
    v = Vector2(2, 2)
    y_axis = Vector2(0, 100)
    actual = v.project(y_axis)
    self.assertEqual(0, actual.x)
    self.assertEqual(v.y, actual.y)