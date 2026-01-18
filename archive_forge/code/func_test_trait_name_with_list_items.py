import unittest
from traits.api import (
def test_trait_name_with_list_items(self):

    class Base(HasTraits):
        pass
    a = Base()
    a.add_trait('pins', List())
    self.assertIn('pins', a.traits())
    self.assertNotIn('pins_items', a.traits())