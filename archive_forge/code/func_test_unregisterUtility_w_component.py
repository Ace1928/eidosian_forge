import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_unregisterUtility_w_component(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.interfaces import Unregistered
    from zope.interface.registry import UtilityRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _name = 'name'
    _to_reg = object()
    comp = self._makeOne()
    comp.registerUtility(_to_reg, ifoo, _name)
    _monkey, _events = self._wrapEvents()
    with _monkey:
        unreg = comp.unregisterUtility(_to_reg, ifoo, _name)
    self.assertTrue(unreg)
    self.assertFalse(comp.utilities._adapters)
    self.assertFalse((ifoo, _name) in comp._utility_registrations)
    self.assertFalse(comp.utilities._subscribers)
    self.assertEqual(len(_events), 1)
    args, kw = _events[0]
    event, = args
    self.assertEqual(kw, {})
    self.assertTrue(isinstance(event, Unregistered))
    self.assertTrue(isinstance(event.object, UtilityRegistration))
    self.assertTrue(event.object.registry is comp)
    self.assertTrue(event.object.provided is ifoo)
    self.assertTrue(event.object.name is _name)
    self.assertTrue(event.object.component is _to_reg)
    self.assertTrue(event.object.factory is None)