import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_None_no_bases_w_factory(self):
    from zope.interface.declarations import objectSpecificationDescriptor

    class Foo:
        __implemented__ = None

        def __call__(self):
            raise NotImplementedError()
    foo = Foo()
    foo.__name__ = 'foo'
    spec = self._callFUT(foo)
    self.assertEqual(spec.__name__, 'zope.interface.tests.test_declarations.foo')
    self.assertIs(spec.inherit, foo)
    self.assertIs(foo.__implemented__, spec)
    self.assertIs(foo.__providedBy__, objectSpecificationDescriptor)
    self.assertNotIn('__provides__', foo.__dict__)