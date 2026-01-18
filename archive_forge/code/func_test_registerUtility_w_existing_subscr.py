import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerUtility_w_existing_subscr(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _info = 'info'
    _name1 = 'name1'
    _name2 = 'name2'
    _to_reg = object()
    comp = self._makeOne()
    comp.registerUtility(_to_reg, ifoo, _name1, _info)
    _monkey, _events = self._wrapEvents()
    with _monkey:
        comp.registerUtility(_to_reg, ifoo, _name2, _info)
    self.assertEqual(comp.utilities._subscribers[0][ifoo][''], (_to_reg,))