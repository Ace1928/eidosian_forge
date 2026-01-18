import unittest
from zope.interface.tests import OptimizationTestMixin
def test_subscribed_non_empty_hit(self):
    IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
    registry = self._makeOne()
    registry.subscribe([IB0], IF0, 'A1')
    self.assertEqual(registry.subscribed([IB0], IF0, 'A1'), 'A1')