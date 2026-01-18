import copy
import pickle
import unittest
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Dict, List, Set, Str, Int, Instance
def test_trait_set_default_kind(self):
    a = A()
    set_trait = a.traits()['aset']
    self.assertEqual(set_trait.default_kind, 'set')