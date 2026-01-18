import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_unregisterSubscriptionAdapter_neither_factory_nor_required(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    comp = self._makeOne()
    self.assertRaises(TypeError, comp.unregisterSubscriptionAdapter, factory=None, provided=ifoo, required=None)