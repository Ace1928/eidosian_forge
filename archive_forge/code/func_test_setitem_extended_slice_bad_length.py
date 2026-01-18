import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_setitem_extended_slice_bad_length(self):
    foo = HasLengthConstrainedLists(at_least_two=[1, 2, 3, 4])
    with self.assertRaises(ValueError):
        foo.at_least_two[1::2] = squares(3)
    self.assertEqual(foo.at_least_two, [1, 2, 3, 4])