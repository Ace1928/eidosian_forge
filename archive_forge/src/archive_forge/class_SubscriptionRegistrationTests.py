import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class SubscriptionRegistrationTests(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.registry import SubscriptionRegistration
        return SubscriptionRegistration

    def _makeOne(self, component=None):
        from zope.interface.declarations import InterfaceClass

        class IFoo(InterfaceClass):
            pass
        ifoo = IFoo('IFoo')
        ibar = IFoo('IBar')

        class _Registry:

            def __repr__(self):
                return '_REGISTRY'
        registry = _Registry()
        name = 'name'
        doc = 'DOCSTRING'
        klass = self._getTargetClass()
        return (klass(registry, (ibar,), ifoo, name, component, doc), registry, name)

    def test_class_conforms_to_ISubscriptionAdapterRegistration(self):
        from zope.interface.interfaces import ISubscriptionAdapterRegistration
        from zope.interface.verify import verifyClass
        verifyClass(ISubscriptionAdapterRegistration, self._getTargetClass())

    def test_instance_conforms_to_ISubscriptionAdapterRegistration(self):
        from zope.interface.interfaces import ISubscriptionAdapterRegistration
        from zope.interface.verify import verifyObject
        sar, _, _ = self._makeOne()
        verifyObject(ISubscriptionAdapterRegistration, sar)