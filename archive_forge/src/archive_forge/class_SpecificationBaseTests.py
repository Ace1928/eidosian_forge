import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class SpecificationBaseTests(GenericSpecificationBaseTests, OptimizationTestMixin):

    def _getTargetClass(self):
        from zope.interface.interface import SpecificationBase
        return SpecificationBase