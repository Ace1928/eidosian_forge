import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Int
from traits.observation._trait_change_event import (
def test_trait_change_event_repr(self):
    event = TraitChangeEvent(object=None, name='name', old=1, new=2)
    actual = repr(event)
    self.assertEqual(actual, "TraitChangeEvent(object=None, name='name', old=1, new=2)")