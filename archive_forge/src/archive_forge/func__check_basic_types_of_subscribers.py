import unittest
from zope.interface.tests import OptimizationTestMixin
def _check_basic_types_of_subscribers(self, registry, expected_order=2):
    self.assertEqual(len(registry._subscribers), expected_order)
    self.assertIsInstance(registry._subscribers, self._getMutableListType())
    MT = self._getMappingType()
    for mapping in registry._subscribers:
        self.assertIsInstance(mapping, MT)
    if expected_order:
        self.assertEqual(registry._subscribers[0], MT())
        self.assertIsInstance(registry._subscribers[1], MT)
        self.assertEqual(len(registry._subscribers[expected_order - 1]), 1)