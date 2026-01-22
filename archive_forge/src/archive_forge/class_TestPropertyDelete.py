import unittest
from traits.api import Any, HasTraits, Int, Property, TraitError
class TestPropertyDelete(unittest.TestCase):

    def test_property_delete(self):
        e = E()
        with self.assertRaises(TraitError):
            del e.a
        with self.assertRaises(TraitError):
            del e.b

    def test_property_reset_traits(self):
        e = E()
        unresetable = e.reset_traits()
        self.assertCountEqual(unresetable, ['a', 'b'])