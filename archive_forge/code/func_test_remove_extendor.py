import unittest
from zope.interface.tests import OptimizationTestMixin
def test_remove_extendor(self):
    from zope.interface import Interface
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))
    registry = self._makeRegistry(IFoo, IBar)
    alb = self._makeOne(registry)
    alb.remove_extendor(IFoo)
    self.assertEqual(sorted(alb._extendors.keys()), sorted([IFoo, IBar, Interface]))
    self.assertEqual(alb._extendors[IFoo], [IBar])
    self.assertEqual(alb._extendors[IBar], [IBar])
    self.assertEqual(sorted(alb._extendors[Interface]), sorted([IBar]))