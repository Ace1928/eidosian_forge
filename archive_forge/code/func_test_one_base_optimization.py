import unittest
def test_one_base_optimization(self):
    c3 = self._makeOne(type(self))
    self.assertIsNotNone(c3._C3__mro)
    c3._merge = None
    self.assertEqual(c3.mro(), list(type(self).__mro__))