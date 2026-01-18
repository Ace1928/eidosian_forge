import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_cached_spec(self):
    from zope.interface import declarations
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    prior = object()

    class Foo:
        pass
    cache = {(Foo, IFoo): prior}
    with _Monkey(declarations, InstanceDeclarations=cache):
        spec = self._callFUT(Foo, IFoo)
    self.assertTrue(spec is prior)