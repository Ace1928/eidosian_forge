import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_wo_declarations_in_class_or_instance(self):

    class Foo:
        pass
    foo = Foo()
    self.assertEqual(list(self._callFUT(foo)), [])