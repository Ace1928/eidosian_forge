import unittest
from zope.interface.tests import OptimizationTestMixin
def test_registered_non_empty_miss(self):
    IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
    registry = self._makeOne()
    registry.register([IB1], None, '', 'A1')
    self.assertEqual(registry.registered([IB2], None, ''), None)