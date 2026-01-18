import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerSubscriptionAdapter_w_nonblank_name(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')
    _name = 'name'
    _info = 'info'

    def _factory(context):
        raise NotImplementedError()
    comp = self._makeOne()
    self.assertRaises(TypeError, comp.registerSubscriptionAdapter, _factory, (ibar,), ifoo, _name, _info)