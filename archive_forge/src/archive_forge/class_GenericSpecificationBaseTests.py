import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class GenericSpecificationBaseTests(unittest.TestCase):

    def _getFallbackClass(self):
        from zope.interface.interface import SpecificationBasePy
        return SpecificationBasePy
    _getTargetClass = _getFallbackClass

    def _makeOne(self):
        return self._getTargetClass()()

    def test_providedBy_miss(self):
        from zope.interface import interface
        from zope.interface.declarations import _empty
        sb = self._makeOne()

        def _providedBy(obj):
            return _empty
        with _Monkey(interface, providedBy=_providedBy):
            self.assertFalse(sb.providedBy(object()))

    def test_implementedBy_miss(self):
        from zope.interface import interface
        from zope.interface.declarations import _empty
        sb = self._makeOne()

        def _implementedBy(obj):
            return _empty
        with _Monkey(interface, implementedBy=_implementedBy):
            self.assertFalse(sb.implementedBy(object()))