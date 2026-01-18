import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_w_optional(self):

    def _func(foo='bar'):
        """DOCSTRING"""
    method = self._callFUT(_func)
    info = method.getSignatureInfo()
    self.assertEqual(list(info['positional']), ['foo'])
    self.assertEqual(list(info['required']), [])
    self.assertEqual(info['optional'], {'foo': 'bar'})
    self.assertEqual(info['varargs'], None)
    self.assertEqual(info['kwargs'], None)