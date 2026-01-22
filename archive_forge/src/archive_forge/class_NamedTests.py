import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class NamedTests(unittest.TestCase):

    def test_class(self):
        from zope.interface.declarations import named

        @named('foo')
        class Foo:
            pass
        self.assertEqual(Foo.__component_name__, 'foo')

    def test_function(self):
        from zope.interface.declarations import named

        @named('foo')
        def doFoo(o):
            raise NotImplementedError()
        self.assertEqual(doFoo.__component_name__, 'foo')

    def test_instance(self):
        from zope.interface.declarations import named

        class Foo:
            pass
        foo = Foo()
        named('foo')(foo)
        self.assertEqual(foo.__component_name__, 'foo')