import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_validateInvariants_inherited_not_called_multiple_times(self):
    _passable_called_with = []

    def _passable(*args, **kw):
        _passable_called_with.append((args, kw))
        return True
    obj = object()
    base = self._makeOne('IBase')
    base.setTaggedValue('invariants', [_passable])
    derived = self._makeOne('IDerived', (base,))
    derived.validateInvariants(obj)
    self.assertEqual(1, len(_passable_called_with))