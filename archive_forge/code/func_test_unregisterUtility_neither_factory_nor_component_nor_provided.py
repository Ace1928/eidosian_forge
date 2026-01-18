import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_unregisterUtility_neither_factory_nor_component_nor_provided(self):
    comp = self._makeOne()
    self.assertRaises(TypeError, comp.unregisterUtility, component=None, provided=None, factory=None)