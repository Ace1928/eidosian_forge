import copy
import pickle
import sys
import unittest
from unittest import mock
from traits.api import HasTraits
from traits.trait_dict_object import TraitDict, TraitDictEvent, TraitDictObject
from traits.trait_errors import TraitError
from traits.trait_types import Dict, Int, Str
def test_update_notifies_with_nonempty_argument(self):
    td = TraitDict({'1': 1, '2': 2}, key_validator=str, notifiers=[self.notification_handler])
    td.update({'1': 1})
    self.assertEqual(td, {'1': 1, '2': 2})
    self.assertEqual(self.added, {})
    self.assertEqual(self.changed, {'1': 1})
    self.assertEqual(self.removed, {})