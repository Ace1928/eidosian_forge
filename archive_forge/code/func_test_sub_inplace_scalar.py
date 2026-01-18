import unittest
from kivy.vector import Vector
from operator import truediv
def test_sub_inplace_scalar(self):
    finalVector = Vector(3, 3)
    finalVector -= 2
    self.assertEqual(finalVector.x, 1)
    self.assertEqual(finalVector.y, 1)