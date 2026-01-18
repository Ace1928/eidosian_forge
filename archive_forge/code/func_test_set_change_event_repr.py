import unittest
from traits.observation._set_change_event import (
from traits.trait_set_object import TraitSet
def test_set_change_event_repr(self):
    event = SetChangeEvent(object=set(), added={1}, removed={3})
    actual = repr(event)
    self.assertEqual(actual, 'SetChangeEvent(object=set(), removed={3}, added={1})')