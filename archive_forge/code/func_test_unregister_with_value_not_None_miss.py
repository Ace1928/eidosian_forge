import unittest
from zope.interface.tests import OptimizationTestMixin
def test_unregister_with_value_not_None_miss(self):
    IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
    registry = self._makeOne()
    orig = object()
    nomatch = object()
    registry.register([IB1], None, '', orig)
    registry.unregister([IB1], None, '', nomatch)
    self.assertIs(registry.registered([IB1], None, ''), orig)