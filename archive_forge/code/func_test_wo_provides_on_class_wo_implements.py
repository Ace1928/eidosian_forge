import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_wo_provides_on_class_wo_implements(self):

    class Foo:
        pass
    foo = Foo()
    spec = self._callFUT(foo)
    self.assertEqual(list(spec), [])