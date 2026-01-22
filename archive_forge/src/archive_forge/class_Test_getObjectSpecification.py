import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_getObjectSpecification(Test_getObjectSpecificationFallback, OptimizationTestMixin):

    def _getTargetClass(self):
        from zope.interface.declarations import getObjectSpecification
        return getObjectSpecification