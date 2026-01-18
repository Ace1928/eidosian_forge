import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_providedBy_invalid_spec_w_provides_no_provides_on_class(self):

    class Foo:
        pass
    foo = Foo()
    foo.__providedBy__ = object()
    expected = foo.__provides__ = object()
    spec = self._callFUT(foo)
    self.assertTrue(spec is expected)