import unittest
from zope.interface.tests import OptimizationTestMixin
def test_unsubscribe_empty(self):
    registry = self._makeOne()
    registry.unsubscribe([None], None, '')
    self.assertEqual(registry.registered([None], None, ''), None)
    self._check_basic_types_of_subscribers(registry, expected_order=0)