import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_queryUtility_hit(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _to_reg = object()
    comp = self._makeOne()
    comp.registerUtility(_to_reg, ifoo)
    self.assertTrue(comp.queryUtility(ifoo) is _to_reg)