import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_pickles_with_utility_registration(self):
    import pickle
    comp = self._makeOne()
    utility = object()
    comp.registerUtility(utility, Interface)
    self.assertIs(utility, comp.getUtility(Interface))
    comp2 = pickle.loads(pickle.dumps(comp))
    self.assertEqual(comp2.__name__, 'test')
    self.assertIsNotNone(comp2.getUtility(Interface))
    comp2.registerUtility(utility, Interface)
    self.assertIs(utility, comp2.getUtility(Interface))
    self._check_equality_after_pickle(comp2)