import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_queryDescriptionFor_miss(self):
    iface = self._makeOne()
    self.assertEqual(iface.queryDescriptionFor('nonesuch'), None)