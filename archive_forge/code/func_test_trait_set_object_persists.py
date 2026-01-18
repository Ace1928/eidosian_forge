import copy
import pickle
import unittest
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Dict, List, Set, Str, Int, Instance
def test_trait_set_object_persists(self):
    a = A()
    set = pickle.loads(pickle.dumps(a.aset))
    self.assertIsNone(set.object())
    set.add(10)
    self.assertEqual(len(a.events), 0)
    a.aset.add(20)
    self.assertEqual(len(a.events), 1)
    set2 = pickle.loads(pickle.dumps(set))
    self.assertIsNone(set2.object())