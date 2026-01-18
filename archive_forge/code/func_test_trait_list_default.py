import copy
import pickle
import unittest
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Dict, List, Set, Str, Int, Instance
def test_trait_list_default(self):
    a = A()
    list_trait = a.traits()['alist']
    self.assertEqual(list_trait.default, [0, 1, 2, 3, 4])
    list_trait.default.append(5)
    self.assertEqual(a.alist, [0, 1, 2, 3, 4])