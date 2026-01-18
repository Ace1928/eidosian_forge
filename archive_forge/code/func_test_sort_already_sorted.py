import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_sort_already_sorted(self):
    tl = TraitList([10, 11, 12, 13, 14], item_validator=int_item_validator, notifiers=[self.notification_handler])
    tl.sort()
    self.assertEqual(tl, [10, 11, 12, 13, 14])
    self.assertEqual(self.index, 0)
    self.assertEqual(self.removed, [10, 11, 12, 13, 14])
    self.assertEqual(self.added, [10, 11, 12, 13, 14])