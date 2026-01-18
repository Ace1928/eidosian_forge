import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_super_when_base_doesnt_implement_interface(self):
    from zope.interface import Interface
    from zope.interface.declarations import implementer

    class IBase(Interface):
        pass

    class IDerived(IBase):
        pass

    class Base:
        pass

    @implementer(IDerived)
    class Derived(Base):
        pass
    derived = Derived()
    self.assertEqual(list(self._callFUT(derived)), [IDerived])
    sup = super(Derived, derived)
    self.assertEqual(list(self._callFUT(sup)), [])