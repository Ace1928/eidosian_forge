import copy
import pickle
import unittest
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Dict, List, Set, Str, Int, Instance
def test_trait_dict_default(self):
    a = A()
    dict_trait = a.traits()['adict']
    self.assertEqual(dict_trait.default, {'a': 1, 'b': 2})
    dict_trait.default.pop('a')
    self.assertEqual(a.adict, {'a': 1, 'b': 2})