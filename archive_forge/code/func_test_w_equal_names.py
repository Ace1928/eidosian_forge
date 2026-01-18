import unittest
from zope.interface import Interface
def test_w_equal_names(self):
    from zope.interface.tests.m1 import I1 as m1_I1
    l = [I1, m1_I1]
    l.sort()
    self.assertEqual(l, [m1_I1, I1])