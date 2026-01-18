import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_wo_provides_on_class_w_implements(self):
    from zope.interface.declarations import implementer
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    @implementer(IFoo)
    class Foo:
        pass
    foo = Foo()
    spec = self._callFUT(foo)
    self.assertEqual(list(spec), [IFoo])