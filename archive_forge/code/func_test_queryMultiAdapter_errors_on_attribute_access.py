import unittest
from zope.interface.tests import OptimizationTestMixin
def test_queryMultiAdapter_errors_on_attribute_access(self):
    from zope.interface.interface import InterfaceClass
    from zope.interface.tests import MissingSomeAttrs
    IFoo = InterfaceClass('IFoo')
    registry = self._makeRegistry()
    alb = self._makeOne(registry)
    alb.lookup = alb._uncached_lookup

    def test(ob):
        return alb.queryMultiAdapter((ob,), IFoo)
    MissingSomeAttrs.test_raises(self, test, expected_missing='__class__')