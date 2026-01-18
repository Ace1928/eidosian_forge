import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_scale_by__subzero(self):
    """The scale method scales around the center of the rectangle"""
    r = Rect(2, 4, 6, 8)
    r.scale_by(0)
    r.scale_by(-1)
    r.scale_by(-1e-06)
    r.scale_by(1e-05)
    rx1 = r.scale_by(10, 1)
    self.assertEqual(r.centerx - r.w * 10 / 2, rx1.x)
    self.assertEqual(r.y, rx1.y)
    self.assertEqual(r.w * 10, rx1.w)
    self.assertEqual(r.h, rx1.h)
    rx2 = r.scale_by(-10, 1)
    self.assertEqual(rx1.x, rx2.x)
    self.assertEqual(rx1.y, rx2.y)
    self.assertEqual(rx1.w, rx2.w)
    self.assertEqual(rx1.h, rx2.h)
    ry1 = r.scale_by(1, 10)
    self.assertEqual(r.x, ry1.x)
    self.assertEqual(r.centery - r.h * 10 / 2, ry1.y)
    self.assertEqual(r.w, ry1.w)
    self.assertEqual(r.h * 10, ry1.h)
    ry2 = r.scale_by(1, -10)
    self.assertEqual(ry1.x, ry2.x)
    self.assertEqual(ry1.y, ry2.y)
    self.assertEqual(ry1.w, ry2.w)
    self.assertEqual(ry1.h, ry2.h)
    r1 = r.scale_by(10)
    self.assertEqual(r.centerx - r.w * 10 / 2, r1.x)
    self.assertEqual(r.centery - r.h * 10 / 2, r1.y)
    self.assertEqual(r.w * 10, r1.w)
    self.assertEqual(r.h * 10, r1.h)