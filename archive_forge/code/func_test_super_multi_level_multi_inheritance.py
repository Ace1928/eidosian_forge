import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_super_multi_level_multi_inheritance(self):
    from zope.interface import Interface
    from zope.interface.declarations import implementer

    class IBase(Interface):
        pass

    class IM1(Interface):
        pass

    class IM2(Interface):
        pass

    class IDerived(IBase):
        pass

    class IUnrelated(Interface):
        pass

    @implementer(IBase)
    class Base:
        pass

    @implementer(IM1)
    class M1(Base):
        pass

    @implementer(IM2)
    class M2(Base):
        pass

    @implementer(IDerived, IUnrelated)
    class Derived(M1, M2):
        pass
    d = Derived()
    sd = super(Derived, d)
    sm1 = super(M1, d)
    sm2 = super(M2, d)
    self.assertEqual(list(self._callFUT(d)), [IDerived, IUnrelated, IM1, IBase, IM2])
    self.assertEqual(list(self._callFUT(sd)), [IM1, IBase, IM2])
    self.assertEqual(list(self._callFUT(sm1)), [IM2, IBase])
    self.assertEqual(list(self._callFUT(sm2)), [IBase])