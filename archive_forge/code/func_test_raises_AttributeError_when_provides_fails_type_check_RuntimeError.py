import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_raises_AttributeError_when_provides_fails_type_check_RuntimeError(self):

    class Foo:
        __provides__ = MissingSomeAttrs(RuntimeError)
    with self.assertRaises(RuntimeError) as exc:
        self._callFUT(Foo())
    self.assertEqual('__class__', exc.exception.args[0])