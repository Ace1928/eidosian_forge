import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerUtility_wo_event(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _info = 'info'
    _name = 'name'
    _to_reg = object()
    comp = self._makeOne()
    _monkey, _events = self._wrapEvents()
    with _monkey:
        comp.registerUtility(_to_reg, ifoo, _name, _info, False)
    self.assertEqual(len(_events), 0)