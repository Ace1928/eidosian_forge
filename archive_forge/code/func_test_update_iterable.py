import copy
import pickle
import sys
import unittest
from unittest import mock
from traits.api import HasTraits
from traits.trait_dict_object import TraitDict, TraitDictEvent, TraitDictObject
from traits.trait_errors import TraitError
from traits.trait_types import Dict, Int, Str
def test_update_iterable(self):
    td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
    td.update([('a', 2), ('b', 4), ('c', 5)])
    self.assertEqual(self.added, {'c': 5})
    self.assertEqual(self.changed, {'a': 1, 'b': 2})
    self.assertEqual(self.removed, {})