import unittest
from kivy.vector import Vector
from operator import truediv
def test_truediv_twovectors(self):
    finalVector = truediv(Vector(6, 6), Vector(2.0, 2.0))
    self.assertAlmostEqual(finalVector.x, 3.0)
    self.assertAlmostEqual(finalVector.y, 3.0)