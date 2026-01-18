import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_unregisterAdapter_wo_explicit_required(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.interfaces import Unregistered
    from zope.interface.registry import AdapterRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')

    class _Factory:
        __component_adapts__ = (ibar,)
    comp = self._makeOne()
    comp.registerAdapter(_Factory, (ibar,), ifoo)
    _monkey, _events = self._wrapEvents()
    with _monkey:
        unreg = comp.unregisterAdapter(_Factory, provided=ifoo)
    self.assertTrue(unreg)
    self.assertEqual(len(_events), 1)
    args, kw = _events[0]
    event, = args
    self.assertEqual(kw, {})
    self.assertTrue(isinstance(event, Unregistered))
    self.assertTrue(isinstance(event.object, AdapterRegistration))
    self.assertTrue(event.object.registry is comp)
    self.assertTrue(event.object.provided is ifoo)
    self.assertEqual(event.object.required, (ibar,))
    self.assertEqual(event.object.name, '')
    self.assertEqual(event.object.info, '')
    self.assertTrue(event.object.factory is _Factory)