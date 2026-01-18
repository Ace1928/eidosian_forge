import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_project_v2_raises_if_other_has_zero_length(self):
    """Check if exception is raise when projected on vector has zero length."""
    v = Vector2(2, 3)
    other = Vector2(0, 0)
    self.assertRaises(ValueError, v.project, other)