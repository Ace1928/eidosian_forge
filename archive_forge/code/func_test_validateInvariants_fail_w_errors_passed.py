import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_validateInvariants_fail_w_errors_passed(self):
    from zope.interface.exceptions import Invalid
    _errors = []
    _fail_called_with = []

    def _fail(*args, **kw):
        _fail_called_with.append((args, kw))
        raise Invalid
    iface = self._makeOne()
    obj = object()
    iface.setTaggedValue('invariants', [_fail])
    self.assertRaises(Invalid, iface.validateInvariants, obj, _errors)
    self.assertEqual(_fail_called_with, [((obj,), {})])
    self.assertEqual(len(_errors), 1)
    self.assertTrue(isinstance(_errors[0], Invalid))