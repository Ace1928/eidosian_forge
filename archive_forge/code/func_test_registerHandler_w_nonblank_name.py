import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerHandler_w_nonblank_name(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _nonblank = 'nonblank'
    comp = self._makeOne()

    def _factory(context):
        raise NotImplementedError()
    self.assertRaises(TypeError, comp.registerHandler, _factory, required=ifoo, name=_nonblank)