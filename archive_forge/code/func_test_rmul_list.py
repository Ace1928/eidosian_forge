import unittest
from kivy.vector import Vector
from operator import truediv
def test_rmul_list(self):
    finalVector = (3, 3) * Vector(2, 2)
    self.assertEqual(finalVector.x, 6)
    self.assertEqual(finalVector.y, 6)