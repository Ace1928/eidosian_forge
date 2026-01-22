import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
class TestTraitListEvent(unittest.TestCase):

    def test_creation(self):
        event = TraitListEvent(index=2, removed=[3], added=[4])
        self.assertEqual(event.index, 2)
        self.assertEqual(event.removed, [3])
        self.assertEqual(event.added, [4])
        event = TraitListEvent(index=2, removed=[3], added=[4])
        self.assertEqual(event.index, 2)
        self.assertEqual(event.removed, [3])
        self.assertEqual(event.added, [4])

    def test_defaults(self):
        event = TraitListEvent()
        self.assertEqual(event.index, 0)
        self.assertEqual(event.removed, [])
        self.assertEqual(event.added, [])

    def test_trait_list_event_str_representation(self):
        """ Test string representation of the TraitListEvent class. """
        desired_repr = 'TraitListEvent(index=0, removed=[], added=[])'
        trait_list_event = TraitListEvent()
        self.assertEqual(desired_repr, str(trait_list_event))
        self.assertEqual(desired_repr, repr(trait_list_event))

    def test_trait_list_event_subclass_str_representation(self):
        """ Test string representation of a subclass of the TraitListEvent
        class. """

        class DifferentName(TraitListEvent):
            pass
        desired_repr = 'DifferentName(index=0, removed=[], added=[])'
        different_name_subclass = DifferentName()
        self.assertEqual(desired_repr, str(different_name_subclass))
        self.assertEqual(desired_repr, repr(different_name_subclass))