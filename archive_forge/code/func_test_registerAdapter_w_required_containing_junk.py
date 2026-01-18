import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerAdapter_w_required_containing_junk(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _info = 'info'
    _name = 'name'

    class _Factory:
        pass
    comp = self._makeOne()
    self.assertRaises(TypeError, comp.registerAdapter, _Factory, [object()], provided=ifoo, name=_name, info=_info)