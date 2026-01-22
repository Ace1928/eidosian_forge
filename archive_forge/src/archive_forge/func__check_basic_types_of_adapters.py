import unittest
from zope.interface.tests import OptimizationTestMixin
def _check_basic_types_of_adapters(self, registry, expected_order=2):
    self.assertEqual(len(registry._adapters), expected_order)
    self.assertIsInstance(registry._adapters, self._getMutableListType())
    MT = self._getMappingType()
    for mapping in registry._adapters:
        self.assertIsInstance(mapping, MT)
    self.assertEqual(registry._adapters[0], MT())
    self.assertIsInstance(registry._adapters[1], MT)
    self.assertEqual(len(registry._adapters[expected_order - 1]), 1)