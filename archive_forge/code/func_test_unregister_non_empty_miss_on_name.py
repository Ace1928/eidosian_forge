import unittest
from zope.interface.tests import OptimizationTestMixin
def test_unregister_non_empty_miss_on_name(self):
    IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
    registry = self._makeOne()
    registry.register([IB1], None, '', 'A1')
    registry.unregister([IB1], None, 'nonesuch')
    self.assertEqual(registry.registered([IB1], None, ''), 'A1')
    self._check_basic_types_of_adapters(registry)
    MT = self._getMappingType()
    self.assertEqual(registry._adapters[1], MT({IB1: MT({None: MT({'': 'A1'})})}))
    PT = self._getProvidedType()
    self.assertEqual(registry._provided, PT({None: 1}))