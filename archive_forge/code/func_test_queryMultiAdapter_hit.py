import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_queryMultiAdapter_hit(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.declarations import implementer

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')
    ibaz = IFoo('IBaz')

    @implementer(ibar)
    class _Context1:
        pass

    @implementer(ibaz)
    class _Context2:
        pass
    _context1 = _Context1()
    _context2 = _Context2()

    class _Factory:

        def __init__(self, context1, context2):
            self.context = (context1, context2)
    comp = self._makeOne()
    comp.registerAdapter(_Factory, (ibar, ibaz), ifoo)
    adapter = comp.queryMultiAdapter((_context1, _context2), ifoo)
    self.assertTrue(isinstance(adapter, _Factory))
    self.assertEqual(adapter.context, (_context1, _context2))