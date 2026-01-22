import unittest
from traits.api import HasTraits, Trait, TraitError, TraitHandler
from traits.trait_base import strx
class StrHandlerCase(unittest.TestCase):

    def test_validator_function(self):
        f = Foo()
        self.assertEqual(f.s, '')
        f.s = 'ok'
        self.assertEqual(f.s, 'ok')
        self.assertRaises(TraitError, setattr, f, 's', 'should fail.')
        self.assertEqual(f.s, 'ok')

    def test_validator_handler(self):
        b = Bar()
        self.assertEqual(b.s, '')
        b.s = 'ok'
        self.assertEqual(b.s, 'ok')
        self.assertRaises(TraitError, setattr, b, 's', 'should fail.')
        self.assertEqual(b.s, 'ok')