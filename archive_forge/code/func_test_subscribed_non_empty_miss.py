import unittest
from zope.interface.tests import OptimizationTestMixin
def test_subscribed_non_empty_miss(self):
    IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
    registry = self._makeOne()
    registry.subscribe([IB1], IF0, 'A1')
    self.assertIsNone(registry.subscribed([IB2], IF0, ''))
    self.assertIsNone(registry.subscribed([IB1], IF1, ''))
    self.assertIsNone(registry.subscribed([IB1], IF0, ''))