import unittest
from kivy.vector import Vector
from operator import truediv
def test_rotate(self):
    v = Vector(100, 0)
    v = v.rotate(45)
    self.assertAlmostEqual(v.x, 70.71067811865476)
    self.assertAlmostEqual(v.y, 70.71067811865474)