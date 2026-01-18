import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_project_v3_raises_if_other_is_not_iterable(self):
    """Check if exception is raise when projected on vector is not iterable."""
    v = Vector3(2, 3, 4)
    other = 10
    self.assertRaises(TypeError, v.project, other)