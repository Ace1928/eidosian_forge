import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class ClassProvidesStrictTests(ClassProvidesTests):

    def _getTargetClass(self):
        ClassProvides = super()._getTargetClass()

        class StrictClassProvides(ClassProvides):

            def _do_calculate_ro(self, base_mros):
                return ClassProvides._do_calculate_ro(self, base_mros=base_mros, strict=True)
        return StrictClassProvides

    def test_overlapping_interfaces_corrected(self):
        from zope.interface import Interface
        from zope.interface import implementedBy
        from zope.interface import implementer

        class IBase(Interface):
            pass

        @implementer(IBase)
        class metaclass(type):
            pass
        cls = metaclass('cls', (object,), {})
        spec = self._makeOne(cls, metaclass, IBase)
        self.assertEqual(spec.__sro__, (spec, implementedBy(metaclass), IBase, implementedBy(type), implementedBy(object), Interface))