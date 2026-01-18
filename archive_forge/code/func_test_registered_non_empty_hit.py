import unittest
from zope.interface.tests import OptimizationTestMixin
def test_registered_non_empty_hit(self):
    registry = self._makeOne()
    registry.register([None], None, '', 'A1')
    self.assertEqual(registry.registered([None], None, ''), 'A1')