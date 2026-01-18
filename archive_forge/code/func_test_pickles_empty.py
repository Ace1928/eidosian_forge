import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_pickles_empty(self):
    import pickle
    comp = self._makeOne()
    pickle.dumps(comp)
    comp2 = pickle.loads(pickle.dumps(comp))
    self.assertEqual(comp2.__name__, 'test')