import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerUtility_w_different_names_same_component(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _info = 'info'
    _name1 = 'name1'
    _name2 = 'name2'
    _other_reg = object()
    _to_reg = object()
    comp = self._makeOne()
    comp.registerUtility(_other_reg, ifoo, _name1, _info)
    _monkey, _events = self._wrapEvents()
    with _monkey:
        comp.registerUtility(_to_reg, ifoo, _name2, _info)
    self.assertEqual(len(_events), 1)
    self.assertEqual(comp._utility_registrations[ifoo, _name1], (_other_reg, _info, None))
    self.assertEqual(comp._utility_registrations[ifoo, _name2], (_to_reg, _info, None))
    self.assertEqual(comp.utilities._subscribers[0][ifoo][''], (_other_reg, _to_reg))