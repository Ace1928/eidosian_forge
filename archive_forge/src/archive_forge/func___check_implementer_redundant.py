import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def __check_implementer_redundant(self, Base):
    Base, IBase = self._check_implementer(Base)

    class Child(Base):
        pass
    returned = self._callFUT(Child, IBase)
    self.assertIn('__implemented__', returned.__dict__)
    self.assertNotIn('__providedBy__', returned.__dict__)
    self.assertIn('__provides__', returned.__dict__)
    spec = Child.__implemented__
    self.assertEqual(spec.declared, ())
    self.assertEqual(spec.inherit, Child)
    self.assertTrue(IBase.providedBy(Child()))