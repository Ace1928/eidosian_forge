import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_w_kwargs(self):

    def _func(**kw):
        """DOCSTRING"""
    method = self._callFUT(_func)
    info = method.getSignatureInfo()
    self.assertEqual(list(info['positional']), [])
    self.assertEqual(list(info['required']), [])
    self.assertEqual(info['optional'], {})
    self.assertEqual(info['varargs'], None)
    self.assertEqual(info['kwargs'], 'kw')