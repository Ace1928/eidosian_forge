import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_queryDescriptionFor_hit(self):
    from zope.interface import Attribute
    ATTRS = {'attr': Attribute('Title', 'Description')}
    iface = self._makeOne(attrs=ATTRS)
    self.assertEqual(iface.queryDescriptionFor('attr'), ATTRS['attr'])