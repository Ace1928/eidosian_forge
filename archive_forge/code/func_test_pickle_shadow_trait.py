import pickle
import unittest
from traits.api import HasTraits, Int, PrefixMap, TraitError
def test_pickle_shadow_trait(self):

    class Person(HasTraits):
        married = PrefixMap({'yes': 1, 'yeah': 1, 'no': 0, 'nah': 0}, default_value='yeah')
    p = Person()
    married_shadow_trait = p.trait('married_')
    reconstituted = pickle.loads(pickle.dumps(married_shadow_trait))
    default_value_callable = reconstituted.default_value()[1]
    self.assertEqual(default_value_callable(p), 1)