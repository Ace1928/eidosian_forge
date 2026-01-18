import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_providedBy_invalid_spec_w_provides_same_provides_on_class(self):
    from zope.interface.declarations import implementer
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    @implementer(IFoo)
    class Foo:
        pass
    foo = Foo()
    foo.__providedBy__ = object()
    foo.__provides__ = Foo.__provides__ = object()
    spec = self._callFUT(foo)
    self.assertEqual(list(spec), [IFoo])