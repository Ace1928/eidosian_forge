import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_set_state_exclude_notifiers(self):
    ts = TraitSet(notifiers=[])
    ts.__setstate__({'notifiers': [self.notification_handler]})
    self.assertEqual(ts.notifiers, [])