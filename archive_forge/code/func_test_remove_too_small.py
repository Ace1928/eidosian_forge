import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_remove_too_small(self):
    foo = HasLengthConstrainedLists(at_least_two=[1, 2])
    with self.assertRaises(TraitError):
        foo.at_least_two.remove(1)
    with self.assertRaises(TraitError):
        foo.at_least_two.remove(2.0)
    with self.assertRaises(TraitError):
        foo.at_least_two.remove(10)
    self.assertEqual(foo.at_least_two, [1, 2])