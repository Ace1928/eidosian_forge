import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerAdapter_with_component_name(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.declarations import named

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')

    @named('foo')
    class Foo:
        pass
    _info = 'info'
    comp = self._makeOne()
    comp.registerAdapter(Foo, (ibar,), ifoo, info=_info)
    self.assertEqual(comp._adapter_registrations[(ibar,), ifoo, 'foo'], (Foo, _info))