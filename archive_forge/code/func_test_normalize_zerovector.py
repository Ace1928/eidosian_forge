import unittest
from kivy.vector import Vector
from operator import truediv
def test_normalize_zerovector(self):
    vector = Vector(0, 0).normalize()
    self.assertEqual(vector.x, 0)
    self.assertEqual(vector.y, 0)
    self.assertEqual(vector.length(), 0)