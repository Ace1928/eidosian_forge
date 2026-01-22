import unittest
class TestC3(unittest.TestCase):

    def _makeOne(self, C, strict=False, base_mros=None):
        from zope.interface.ro import C3
        return C3.resolver(C, strict, base_mros)

    def test_base_mros_given(self):
        c3 = self._makeOne(type(self), base_mros={unittest.TestCase: unittest.TestCase.__mro__})
        memo = c3.memo
        self.assertIn(unittest.TestCase, memo)
        self.assertIsNone(memo[unittest.TestCase].had_inconsistency)

    def test_one_base_optimization(self):
        c3 = self._makeOne(type(self))
        self.assertIsNotNone(c3._C3__mro)
        c3._merge = None
        self.assertEqual(c3.mro(), list(type(self).__mro__))