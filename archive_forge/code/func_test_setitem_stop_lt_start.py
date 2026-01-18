import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_setitem_stop_lt_start(self):
    events = []
    foo = HasLengthConstrainedLists(at_least_two=[1, 2, 3, 4])
    foo.on_trait_change(lambda event: events.append(event), 'at_least_two_items')
    foo.at_least_two[4:2] = [5, 6, 7]
    self.assertEqual(len(events), 1)
    event = events[0]
    self.assertEqual(event.index, 4)
    self.assertEqual(event.removed, [])
    self.assertEqual(event.added, [5, 6, 7])