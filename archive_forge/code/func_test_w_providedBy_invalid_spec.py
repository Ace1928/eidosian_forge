import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_providedBy_invalid_spec(self):

    class Foo:
        pass
    foo = Foo()
    foo.__providedBy__ = object()
    spec = self._callFUT(foo)
    self.assertEqual(list(spec), [])