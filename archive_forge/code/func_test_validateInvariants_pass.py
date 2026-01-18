import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_validateInvariants_pass(self):
    _called_with = []

    def _passable(*args, **kw):
        _called_with.append((args, kw))
        return True
    iface = self._makeOne()
    obj = object()
    iface.setTaggedValue('invariants', [_passable])
    self.assertEqual(iface.validateInvariants(obj), None)
    self.assertEqual(_called_with, [((obj,), {})])