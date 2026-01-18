import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_unregisterSubscriptionAdapter_neither_factory_nor_provided(self):
    comp = self._makeOne()
    self.assertRaises(TypeError, comp.unregisterSubscriptionAdapter, factory=None, provided=None)