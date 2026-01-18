import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registeredUtilities_notempty(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.registry import UtilityRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _info = 'info'
    _name1 = 'name1'
    _name2 = 'name2'
    _to_reg = object()
    comp = self._makeOne()
    comp.registerUtility(_to_reg, ifoo, _name1, _info)
    comp.registerUtility(_to_reg, ifoo, _name2, _info)
    reg = sorted(comp.registeredUtilities(), key=lambda r: r.name)
    self.assertEqual(len(reg), 2)
    self.assertTrue(isinstance(reg[0], UtilityRegistration))
    self.assertTrue(reg[0].registry is comp)
    self.assertTrue(reg[0].provided is ifoo)
    self.assertTrue(reg[0].name is _name1)
    self.assertTrue(reg[0].component is _to_reg)
    self.assertTrue(reg[0].info is _info)
    self.assertTrue(reg[0].factory is None)
    self.assertTrue(isinstance(reg[1], UtilityRegistration))
    self.assertTrue(reg[1].registry is comp)
    self.assertTrue(reg[1].provided is ifoo)
    self.assertTrue(reg[1].name is _name2)
    self.assertTrue(reg[1].component is _to_reg)
    self.assertTrue(reg[1].info is _info)
    self.assertTrue(reg[1].factory is None)