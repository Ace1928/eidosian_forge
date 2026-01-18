import unittest
from zope.interface.tests import OptimizationTestMixin
def test_registered_empty(self):
    registry = self._makeOne()
    self.assertEqual(registry.registered([None], None, ''), None)
    registered = list(registry.allRegistrations())
    self.assertEqual(registered, [])