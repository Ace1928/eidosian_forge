import pickle
import unittest
from traits.api import HasTraits, TraitError, TraitPrefixList, Trait
class TestTraitPrefixList(unittest.TestCase):

    def test_pickle_roundtrip(self):
        with self.assertWarns(DeprecationWarning):

            class A(HasTraits):
                foo = Trait('one', TraitPrefixList('zero', 'one', 'two'))
        a = A()
        foo_trait = a.traits()['foo']
        reconstituted = pickle.loads(pickle.dumps(foo_trait))
        self.assertEqual(foo_trait.validate(a, 'foo', 'ze'), 'zero')
        with self.assertRaises(TraitError):
            foo_trait.validate(a, 'foo', 'zero-knowledge')
        self.assertEqual(reconstituted.validate(a, 'foo', 'ze'), 'zero')
        with self.assertRaises(TraitError):
            reconstituted.validate(a, 'foo', 'zero-knowledge')