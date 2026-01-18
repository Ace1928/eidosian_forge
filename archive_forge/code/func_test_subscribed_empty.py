import unittest
from zope.interface.tests import OptimizationTestMixin
def test_subscribed_empty(self):
    registry = self._makeOne()
    self.assertIsNone(registry.subscribed([None], None, ''))
    subscribed = list(registry.allSubscriptions())
    self.assertEqual(subscribed, [])