import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class HandlerRegistrationTests(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.registry import HandlerRegistration
        return HandlerRegistration

    def _makeOne(self, component=None):
        from zope.interface.declarations import InterfaceClass

        class IFoo(InterfaceClass):
            pass
        ifoo = IFoo('IFoo')

        class _Registry:

            def __repr__(self):
                return '_REGISTRY'
        registry = _Registry()
        name = 'name'
        doc = 'DOCSTRING'
        klass = self._getTargetClass()
        return (klass(registry, (ifoo,), name, component, doc), registry, name)

    def test_class_conforms_to_IHandlerRegistration(self):
        from zope.interface.interfaces import IHandlerRegistration
        from zope.interface.verify import verifyClass
        verifyClass(IHandlerRegistration, self._getTargetClass())

    def test_instance_conforms_to_IHandlerRegistration(self):
        from zope.interface.interfaces import IHandlerRegistration
        from zope.interface.verify import verifyObject
        hr, _, _ = self._makeOne()
        verifyObject(IHandlerRegistration, hr)

    def test_properties(self):

        def _factory(context):
            raise NotImplementedError()
        hr, _, _ = self._makeOne(_factory)
        self.assertTrue(hr.handler is _factory)
        self.assertTrue(hr.factory is hr.handler)
        self.assertTrue(hr.provided is None)

    def test___repr___factory_w_name(self):

        class _Factory:
            __name__ = 'TEST'
        hr, _registry, _name = self._makeOne(_Factory())
        self.assertEqual(repr(hr), ('HandlerRegistration(_REGISTRY, [IFoo], %r, TEST, ' + "'DOCSTRING')") % _name)

    def test___repr___factory_wo_name(self):

        class _Factory:

            def __repr__(self):
                return 'TEST'
        hr, _registry, _name = self._makeOne(_Factory())
        self.assertEqual(repr(hr), ('HandlerRegistration(_REGISTRY, [IFoo], %r, TEST, ' + "'DOCSTRING')") % _name)