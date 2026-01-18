import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_existing_Implements(self):
    from zope.interface.declarations import Implements
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar')
    impl = Implements(IFoo)
    impl.declared = (IFoo,)

    class Foo:
        __implemented__ = impl
    impl.inherit = Foo
    self._callFUT(Foo, IBar)
    self.assertIs(Foo.__implemented__, impl)
    self.assertEqual(impl.inherit, Foo)
    self.assertEqual(impl.declared, self._order_for_two(IFoo, IBar))