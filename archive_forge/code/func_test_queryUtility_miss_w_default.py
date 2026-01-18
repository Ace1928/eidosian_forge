import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_queryUtility_miss_w_default(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    comp = self._makeOne()
    _default = object()
    self.assertTrue(comp.queryUtility(ifoo, default=_default) is _default)