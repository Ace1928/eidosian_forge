import copy
import pickle
import sys
import unittest
from unittest import mock
from traits.api import HasTraits
from traits.trait_dict_object import TraitDict, TraitDictEvent, TraitDictObject
from traits.trait_errors import TraitError
from traits.trait_types import Dict, Int, Str
def test_setdefault_with_casting(self):
    notifier = mock.Mock()
    td = TraitDict(key_validator=str, value_validator=str, notifiers=[notifier, self.notification_handler])
    td.setdefault(1, 2)
    self.assertEqual(td, {'1': '2'})
    self.assertEqual(notifier.call_count, 1)
    self.assertEqual(self.removed, {})
    self.assertEqual(self.added, {'1': '2'})
    self.assertEqual(self.changed, {})
    notifier.reset_mock()
    td.setdefault(1, 4)
    self.assertEqual(td, {'1': '4'})
    self.assertEqual(notifier.call_count, 1)
    self.assertEqual(self.removed, {})
    self.assertEqual(self.added, {})
    self.assertEqual(self.changed, {'1': '2'})