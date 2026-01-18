import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerUtility_replaces_existing_reg(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.interfaces import Registered
    from zope.interface.interfaces import Unregistered
    from zope.interface.registry import UtilityRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _info = 'info'
    _name = 'name'
    _before, _after = (object(), object())
    comp = self._makeOne()
    comp.registerUtility(_before, ifoo, _name, _info)
    _monkey, _events = self._wrapEvents()
    with _monkey:
        comp.registerUtility(_after, ifoo, _name, _info)
    self.assertEqual(len(_events), 2)
    args, kw = _events[0]
    event, = args
    self.assertEqual(kw, {})
    self.assertTrue(isinstance(event, Unregistered))
    self.assertTrue(isinstance(event.object, UtilityRegistration))
    self.assertTrue(event.object.registry is comp)
    self.assertTrue(event.object.provided is ifoo)
    self.assertTrue(event.object.name is _name)
    self.assertTrue(event.object.component is _before)
    self.assertTrue(event.object.info is _info)
    self.assertTrue(event.object.factory is None)
    args, kw = _events[1]
    event, = args
    self.assertEqual(kw, {})
    self.assertTrue(isinstance(event, Registered))
    self.assertTrue(isinstance(event.object, UtilityRegistration))
    self.assertTrue(event.object.registry is comp)
    self.assertTrue(event.object.provided is ifoo)
    self.assertTrue(event.object.name is _name)
    self.assertTrue(event.object.component is _after)
    self.assertTrue(event.object.info is _info)
    self.assertTrue(event.object.factory is None)