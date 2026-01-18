import unittest
from zope.interface.tests import OptimizationTestMixin
def test_unsubscribe_non_empty_miss_on_required(self):
    IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
    registry = self._makeOne()
    registry.subscribe([IB1], None, 'A1')
    self._check_basic_types_of_subscribers(registry, expected_order=2)
    registry.unsubscribe([IB2], None, '')
    self.assertEqual(len(registry._subscribers), 2)
    MT = self._getMappingType()
    L = self._getLeafSequenceType()
    self.assertEqual(registry._subscribers[1], MT({IB1: MT({None: MT({'': L(('A1',))})})}))