import copy
import pickle
import unittest
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Dict, List, Set, Str, Int, Instance
def test_trait_list_object_copies(self):
    a = A()
    list = copy.deepcopy(a.alist)
    self.assertIsNone(list.object())
    list.append(10)
    self.assertEqual(len(a.events), 0)
    a.alist.append(20)
    self.assertEqual(len(a.events), 1)
    list2 = copy.deepcopy(list)
    list2.append(30)
    self.assertIsNone(list2.object())