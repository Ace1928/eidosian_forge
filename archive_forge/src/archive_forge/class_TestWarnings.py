import sys
import unittest
import warnings
class TestWarnings(unittest.TestCase):

    def test_assert(self):
        self.assertEquals(2 + 2, 4)
        self.assertEquals(2 * 2, 4)
        self.assertEquals(2 ** 2, 4)

    def test_fail(self):
        self.failUnless(1)
        self.failUnless(True)

    def test_other_unittest(self):
        self.assertAlmostEqual(2 + 2, 4)
        self.assertNotAlmostEqual(4 + 4, 2)

    def test_deprecation(self):
        warnings.warn('dw', DeprecationWarning)
        warnings.warn('dw', DeprecationWarning)
        warnings.warn('dw', DeprecationWarning)

    def test_import(self):
        warnings.warn('iw', ImportWarning)
        warnings.warn('iw', ImportWarning)
        warnings.warn('iw', ImportWarning)

    def test_warning(self):
        warnings.warn('uw')
        warnings.warn('uw')
        warnings.warn('uw')

    def test_function(self):
        warnfun()
        warnfun()
        warnfun()