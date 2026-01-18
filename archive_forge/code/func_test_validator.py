import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_validator(self):
    ts = TraitSet({1, 2, 3}, item_validator=int_validator)
    self.assertEqual(ts, {1, 2, 3})
    self.assertEqual(ts.item_validator, int_validator)
    self.assertEqual(ts.notifiers, [])