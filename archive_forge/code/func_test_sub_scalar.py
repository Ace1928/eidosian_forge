import unittest
from kivy.vector import Vector
from operator import truediv
def test_sub_scalar(self):
    with self.assertRaises(TypeError):
        Vector(3, 3) - 2