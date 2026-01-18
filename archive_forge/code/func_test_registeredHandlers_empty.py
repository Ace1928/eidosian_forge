import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registeredHandlers_empty(self):
    comp = self._makeOne()
    self.assertFalse(list(comp.registeredHandlers()))