import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_iface_implemented_by_class(self):
    from zope.interface.declarations import implementer
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    @implementer(IFoo)
    class Foo:
        pass
    obj = Foo()
    self.assertRaises(ValueError, self._callFUT, obj, IFoo)