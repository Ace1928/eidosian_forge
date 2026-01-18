import unittest
from traits.api import Any, HasTraits, Int, Property, TraitError
def test_property_reset_traits(self):
    e = E()
    unresetable = e.reset_traits()
    self.assertCountEqual(unresetable, ['a', 'b'])