import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registeredSubscriptionAdapters_notempty(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.registry import SubscriptionRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IFoo')
    _info = 'info'
    _blank = ''

    class _Factory:
        pass
    comp = self._makeOne()
    comp.registerSubscriptionAdapter(_Factory, (ibar,), ifoo, info=_info)
    comp.registerSubscriptionAdapter(_Factory, (ibar,), ifoo, info=_info)
    reg = list(comp.registeredSubscriptionAdapters())
    self.assertEqual(len(reg), 2)
    self.assertTrue(isinstance(reg[0], SubscriptionRegistration))
    self.assertTrue(reg[0].registry is comp)
    self.assertTrue(reg[0].provided is ifoo)
    self.assertEqual(reg[0].required, (ibar,))
    self.assertEqual(reg[0].name, _blank)
    self.assertTrue(reg[0].info is _info)
    self.assertTrue(reg[0].factory is _Factory)
    self.assertTrue(isinstance(reg[1], SubscriptionRegistration))
    self.assertTrue(reg[1].registry is comp)
    self.assertTrue(reg[1].provided is ifoo)
    self.assertEqual(reg[1].required, (ibar,))
    self.assertEqual(reg[1].name, _blank)
    self.assertTrue(reg[1].info is _info)
    self.assertTrue(reg[1].factory is _Factory)