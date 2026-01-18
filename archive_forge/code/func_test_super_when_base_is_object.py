import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_super_when_base_is_object(self):
    from zope.interface import Interface
    from zope.interface.declarations import implementer

    class IBase(Interface):
        pass

    class IDerived(IBase):
        pass

    @implementer(IDerived)
    class Derived:
        pass
    derived = Derived()
    self.assertEqual(list(self._callFUT(derived)), [IDerived])
    sup = super(Derived, derived)
    fut = self._callFUT(sup)
    self.assertIsNone(fut._dependents)
    self.assertEqual(list(fut), [])