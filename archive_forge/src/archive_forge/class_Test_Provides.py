import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_Provides(unittest.TestCase):

    def _callFUT(self, *args, **kw):
        from zope.interface.declarations import Provides
        return Provides(*args, **kw)

    def test_no_cached_spec(self):
        from zope.interface import declarations
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        cache = {}

        class Foo:
            pass
        with _Monkey(declarations, InstanceDeclarations=cache):
            spec = self._callFUT(Foo, IFoo)
        self.assertEqual(list(spec), [IFoo])
        self.assertTrue(cache[Foo, IFoo] is spec)

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