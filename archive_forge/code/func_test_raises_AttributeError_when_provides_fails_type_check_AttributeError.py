import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_raises_AttributeError_when_provides_fails_type_check_AttributeError(self):

    class Foo:
        __provides__ = MissingSomeAttrs(AttributeError)
    self._callFUT(Foo())