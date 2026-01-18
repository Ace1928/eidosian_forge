import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_trait_list_event_str_representation(self):
    """ Test string representation of the TraitListEvent class. """
    desired_repr = 'TraitListEvent(index=0, removed=[], added=[])'
    trait_list_event = TraitListEvent()
    self.assertEqual(desired_repr, str(trait_list_event))
    self.assertEqual(desired_repr, repr(trait_list_event))