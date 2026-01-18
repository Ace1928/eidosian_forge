import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_unregisterSubscriptionAdapter_w_nonblank_name(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')
    _nonblank = 'nonblank'
    comp = self._makeOne()
    self.assertRaises(TypeError, comp.unregisterSubscriptionAdapter, required=ifoo, provided=ibar, name=_nonblank)