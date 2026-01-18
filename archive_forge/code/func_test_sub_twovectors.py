import unittest
from kivy.vector import Vector
from operator import truediv
def test_sub_twovectors(self):
    finalVector = Vector(3, 3) - Vector(2, 2)
    self.assertEqual(finalVector.x, 1)
    self.assertEqual(finalVector.y, 1)