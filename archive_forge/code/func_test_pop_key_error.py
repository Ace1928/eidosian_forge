import copy
import pickle
import sys
import unittest
from unittest import mock
from traits.api import HasTraits
from traits.trait_dict_object import TraitDict, TraitDictEvent, TraitDictObject
from traits.trait_errors import TraitError
from traits.trait_types import Dict, Int, Str
def test_pop_key_error(self):
    python_dict = {}
    with self.assertRaises(KeyError) as python_e:
        python_dict.pop('a')
    td = TraitDict()
    with self.assertRaises(KeyError) as trait_e:
        td.pop('a')
    self.assertEqual(str(trait_e.exception), str(python_e.exception))