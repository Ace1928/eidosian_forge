import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_unregisterSubscriptionAdapter_hit_w_factory(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.interfaces import Unregistered
    from zope.interface.registry import SubscriptionRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')

    class _Factory:
        pass
    comp = self._makeOne()
    comp.registerSubscriptionAdapter(_Factory, (ibar,), ifoo)
    _monkey, _events = self._wrapEvents()
    with _monkey:
        unreg = comp.unregisterSubscriptionAdapter(_Factory, (ibar,), ifoo)
    self.assertTrue(unreg)
    self.assertFalse(comp.adapters._subscribers)
    self.assertFalse(comp._subscription_registrations)
    self.assertEqual(len(_events), 1)
    args, kw = _events[0]
    event, = args
    self.assertEqual(kw, {})
    self.assertTrue(isinstance(event, Unregistered))
    self.assertTrue(isinstance(event.object, SubscriptionRegistration))
    self.assertTrue(event.object.registry is comp)
    self.assertTrue(event.object.provided is ifoo)
    self.assertEqual(event.object.required, (ibar,))
    self.assertEqual(event.object.name, '')
    self.assertEqual(event.object.info, '')
    self.assertTrue(event.object.factory is _Factory)