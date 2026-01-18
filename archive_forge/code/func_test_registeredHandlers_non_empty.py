import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registeredHandlers_non_empty(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.registry import HandlerRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')

    def _factory1(context):
        raise NotImplementedError()

    def _factory2(context):
        raise NotImplementedError()
    comp = self._makeOne()
    comp.registerHandler(_factory1, (ifoo,))
    comp.registerHandler(_factory2, (ifoo,))

    def _factory_name(x):
        return x.factory.__code__.co_name
    subscribers = sorted(comp.registeredHandlers(), key=_factory_name)
    self.assertEqual(len(subscribers), 2)
    self.assertTrue(isinstance(subscribers[0], HandlerRegistration))
    self.assertEqual(subscribers[0].required, (ifoo,))
    self.assertEqual(subscribers[0].name, '')
    self.assertEqual(subscribers[0].factory, _factory1)
    self.assertEqual(subscribers[0].info, '')
    self.assertTrue(isinstance(subscribers[1], HandlerRegistration))
    self.assertEqual(subscribers[1].required, (ifoo,))
    self.assertEqual(subscribers[1].name, '')
    self.assertEqual(subscribers[1].factory, _factory2)
    self.assertEqual(subscribers[1].info, '')