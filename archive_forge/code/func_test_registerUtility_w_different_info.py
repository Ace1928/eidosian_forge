import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerUtility_w_different_info(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _info1 = 'info1'
    _info2 = 'info2'
    _name = 'name'
    _to_reg = object()
    comp = self._makeOne()
    comp.registerUtility(_to_reg, ifoo, _name, _info1)
    _monkey, _events = self._wrapEvents()
    with _monkey:
        comp.registerUtility(_to_reg, ifoo, _name, _info2)
    self.assertEqual(len(_events), 2)
    self.assertEqual(comp._utility_registrations[ifoo, _name], (_to_reg, _info2, None))
    self.assertEqual(comp.utilities._subscribers[0][ifoo][''], (_to_reg,))