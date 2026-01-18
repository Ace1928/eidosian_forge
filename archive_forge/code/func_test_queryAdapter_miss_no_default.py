import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_queryAdapter_miss_no_default(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    comp = self._makeOne()
    _context = object()
    self.assertTrue(comp.queryAdapter(_context, ifoo) is None)