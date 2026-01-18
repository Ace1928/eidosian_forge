import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_subscribers_empty(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.declarations import implementer

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')
    comp = self._makeOne()

    @implementer(ibar)
    class Bar:
        pass
    bar = Bar()
    self.assertEqual(list(comp.subscribers((bar,), ifoo)), [])