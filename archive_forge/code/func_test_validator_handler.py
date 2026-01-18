import unittest
from traits.api import HasTraits, Trait, TraitError, TraitHandler
from traits.trait_base import strx
def test_validator_handler(self):
    b = Bar()
    self.assertEqual(b.s, '')
    b.s = 'ok'
    self.assertEqual(b.s, 'ok')
    self.assertRaises(TraitError, setattr, b, 's', 'should fail.')
    self.assertEqual(b.s, 'ok')