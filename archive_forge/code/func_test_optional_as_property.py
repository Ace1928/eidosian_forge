import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_optional_as_property(self):
    method = self._makeOne()
    self.assertEqual(method.optional, {})
    method.optional = {'foo': 'bar'}
    self.assertEqual(method.optional, {'foo': 'bar'})
    del method.optional
    self.assertEqual(method.optional, {})