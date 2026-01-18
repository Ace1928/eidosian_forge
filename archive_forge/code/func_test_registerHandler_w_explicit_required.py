import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerHandler_w_explicit_required(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.interfaces import Registered
    from zope.interface.registry import HandlerRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _blank = ''
    _info = 'info'

    def _factory(context):
        raise NotImplementedError()
    comp = self._makeOne()
    _monkey, _events = self._wrapEvents()
    with _monkey:
        comp.registerHandler(_factory, (ifoo,), info=_info)
    reg = comp.adapters._subscribers[1][ifoo][None][_blank]
    self.assertEqual(len(reg), 1)
    self.assertTrue(reg[0] is _factory)
    self.assertEqual(comp._handler_registrations, [((ifoo,), _blank, _factory, _info)])
    self.assertEqual(len(_events), 1)
    args, kw = _events[0]
    event, = args
    self.assertEqual(kw, {})
    self.assertTrue(isinstance(event, Registered))
    self.assertTrue(isinstance(event.object, HandlerRegistration))
    self.assertTrue(event.object.registry is comp)
    self.assertEqual(event.object.required, (ifoo,))
    self.assertEqual(event.object.name, _blank)
    self.assertTrue(event.object.info is _info)
    self.assertTrue(event.object.factory is _factory)