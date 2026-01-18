import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_simple_metaclass(self):
    from zope.interface.declarations import implementer
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar')

    @implementer(IFoo)
    class Foo:
        pass
    cp = Foo.__provides__ = self._makeOne(Foo, type(Foo), IBar)
    self.assertTrue(Foo.__provides__ is cp)
    self.assertEqual(list(Foo().__provides__), [IFoo])