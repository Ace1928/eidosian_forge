import unittest
from zope.interface.tests import OptimizationTestMixin
class VerifyingBaseTests(VerifyingBaseFallbackTests, OptimizationTestMixin):

    def _getTargetClass(self):
        from zope.interface.adapter import VerifyingBase
        return VerifyingBase