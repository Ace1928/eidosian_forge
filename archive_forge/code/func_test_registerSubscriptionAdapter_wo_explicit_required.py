import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerSubscriptionAdapter_wo_explicit_required(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.interfaces import Registered
    from zope.interface.registry import SubscriptionRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')
    _info = 'info'
    _blank = ''

    class _Factory:
        __component_adapts__ = (ibar,)
    comp = self._makeOne()
    _monkey, _events = self._wrapEvents()
    with _monkey:
        comp.registerSubscriptionAdapter(_Factory, provided=ifoo, info=_info)
    reg = comp.adapters._subscribers[1][ibar][ifoo][_blank]
    self.assertEqual(len(reg), 1)
    self.assertTrue(reg[0] is _Factory)
    self.assertEqual(comp._subscription_registrations, [((ibar,), ifoo, _blank, _Factory, _info)])
    self.assertEqual(len(_events), 1)
    args, kw = _events[0]
    event, = args
    self.assertEqual(kw, {})
    self.assertTrue(isinstance(event, Registered))
    self.assertTrue(isinstance(event.object, SubscriptionRegistration))
    self.assertTrue(event.object.registry is comp)
    self.assertTrue(event.object.provided is ifoo)
    self.assertEqual(event.object.required, (ibar,))
    self.assertEqual(event.object.name, _blank)
    self.assertTrue(event.object.info is _info)
    self.assertTrue(event.object.factory is _Factory)