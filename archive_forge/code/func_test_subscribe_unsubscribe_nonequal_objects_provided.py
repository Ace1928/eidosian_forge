import unittest
from zope.interface.tests import OptimizationTestMixin
def test_subscribe_unsubscribe_nonequal_objects_provided(self):
    IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
    registry = self._makeOne()
    first = object()
    second = object()
    registry.subscribe([IB1], IR0, first)
    registry.subscribe([IB1], IR0, second)
    MT = self._getMappingType()
    L = self._getLeafSequenceType()
    PT = self._getProvidedType()
    self.assertEqual(registry._subscribers[1], MT({IB1: MT({IR0: MT({'': L((first, second))})})}))
    self.assertEqual(registry._provided, PT({IR0: 2}))
    registry.unsubscribe([IB1], IR0, first)
    registry.unsubscribe([IB1], IR0, second)
    self.assertEqual(len(registry._subscribers), 0)
    self.assertEqual(registry._provided, PT())