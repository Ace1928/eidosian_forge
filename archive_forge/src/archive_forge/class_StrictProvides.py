import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class StrictProvides(ProvidesClass):

    def _do_calculate_ro(self, base_mros):
        return ProvidesClass._do_calculate_ro(self, base_mros=base_mros, strict=True)