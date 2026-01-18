import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_super_when_object_directly_provides(self):
    from zope.interface import Interface
    from zope.interface.declarations import directlyProvides
    from zope.interface.declarations import implementer

    class IBase(Interface):
        pass

    class IDerived(IBase):
        pass

    @implementer(IBase)
    class Base:
        pass

    class Derived(Base):
        pass
    derived = Derived()
    self.assertEqual(list(self._callFUT(derived)), [IBase])
    directlyProvides(derived, IDerived)
    self.assertEqual(list(self._callFUT(derived)), [IDerived, IBase])
    sup = super(Derived, derived)
    fut = self._callFUT(sup)
    self.assertIsNone(fut._dependents)
    self.assertEqual(list(fut), [IBase])