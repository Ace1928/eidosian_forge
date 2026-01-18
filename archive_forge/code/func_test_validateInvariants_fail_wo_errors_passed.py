import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_validateInvariants_fail_wo_errors_passed(self):
    from zope.interface.exceptions import Invalid
    _passable_called_with = []

    def _passable(*args, **kw):
        _passable_called_with.append((args, kw))
        return True
    _fail_called_with = []

    def _fail(*args, **kw):
        _fail_called_with.append((args, kw))
        raise Invalid
    iface = self._makeOne()
    obj = object()
    iface.setTaggedValue('invariants', [_passable, _fail])
    self.assertRaises(Invalid, iface.validateInvariants, obj)
    self.assertEqual(_passable_called_with, [((obj,), {})])
    self.assertEqual(_fail_called_with, [((obj,), {})])