import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_w_name(self):

    def _func():
        """DOCSTRING"""
    method = self._callFUT(_func, name='anotherName')
    self.assertEqual(method.getName(), 'anotherName')