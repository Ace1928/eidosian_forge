import unittest
from zope.interface.tests import OptimizationTestMixin
def test_unsubscribe_hit(self):
    IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
    registry = self._makeOne()
    orig = object()
    registry.subscribe([IB1], None, orig)
    MT = self._getMappingType()
    L = self._getLeafSequenceType()
    PT = self._getProvidedType()
    self._check_basic_types_of_subscribers(registry)
    self.assertEqual(registry._subscribers[1], MT({IB1: MT({None: MT({'': L((orig,))})})}))
    self.assertEqual(registry._provided, PT({}))
    registry.unsubscribe([IB1], None, orig)
    self.assertEqual(len(registry._subscribers), 0)
    self.assertEqual(registry._provided, PT({}))