import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerUtility_w_factory(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.interfaces import Registered
    from zope.interface.registry import UtilityRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _info = 'info'
    _name = 'name'
    _to_reg = object()

    def _factory():
        return _to_reg
    comp = self._makeOne()
    _monkey, _events = self._wrapEvents()
    with _monkey:
        comp.registerUtility(None, ifoo, _name, _info, factory=_factory)
    self.assertEqual(len(_events), 1)
    args, kw = _events[0]
    event, = args
    self.assertEqual(kw, {})
    self.assertTrue(isinstance(event, Registered))
    self.assertTrue(isinstance(event.object, UtilityRegistration))
    self.assertTrue(event.object.registry is comp)
    self.assertTrue(event.object.provided is ifoo)
    self.assertTrue(event.object.name is _name)
    self.assertTrue(event.object.component is _to_reg)
    self.assertTrue(event.object.info is _info)
    self.assertTrue(event.object.factory is _factory)