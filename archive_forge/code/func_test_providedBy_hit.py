import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_providedBy_hit(self):
    from zope.interface import interface
    sb = self._makeOne()

    class _Decl:
        _implied = {sb: {}}

    def _providedBy(obj):
        return _Decl()
    with _Monkey(interface, providedBy=_providedBy):
        self.assertTrue(sb.providedBy(object()))