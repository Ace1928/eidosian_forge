import copy
import pickle
import sys
import unittest
from unittest import mock
from traits.api import HasTraits
from traits.trait_dict_object import TraitDict, TraitDictEvent, TraitDictObject
from traits.trait_errors import TraitError
from traits.trait_types import Dict, Int, Str
def test_update_with_empty_argument(self):
    td = TraitDict({'1': 1, '2': 2}, key_validator=str, notifiers=[self.notification_handler])
    td.update([])
    td.update({})
    self.assertEqual(td, {'1': 1, '2': 2})
    self.assertIsNone(self.added)
    self.assertIsNone(self.changed)
    self.assertIsNone(self.removed)