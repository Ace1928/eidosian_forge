import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_unregisterAdapter_miss(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')

    class _Factory:
        pass
    comp = self._makeOne()
    _monkey, _events = self._wrapEvents()
    with _monkey:
        unreg = comp.unregisterAdapter(_Factory, (ibar,), ifoo)
    self.assertFalse(unreg)