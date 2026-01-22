import copy
import pickle
import sys
import unittest
from unittest import mock
from traits.api import HasTraits
from traits.trait_dict_object import TraitDict, TraitDictEvent, TraitDictObject
from traits.trait_errors import TraitError
from traits.trait_types import Dict, Int, Str
class TestTraitDictObject(unittest.TestCase):
    """ Test TraitDictObject operations."""

    class TestClass(HasTraits):
        dict_1 = Dict(Str)
        dict_2 = Dict(Int, Str)

    def test_trait_dict_object_validate_key(self):
        obj = TestTraitDictObject.TestClass()
        trait_dict_obj = TraitDictObject(trait=obj.trait('dict_1').trait_type, object=obj, name='a', value={})
        trait_dict_obj.key_validator('1')
        with self.assertRaises(TraitError):
            trait_dict_obj.key_validator(1)

    def test_trait_dict_object_validate_value(self):
        obj = TestTraitDictObject.TestClass()
        trait_dict_obj = TraitDictObject(trait=obj.trait('dict_2').trait_type, object=obj, name='a', value={})
        trait_dict_obj.value_validator('1')
        with self.assertRaises(TraitError):
            trait_dict_obj.value_validator(1)

    def test_trait_dict_object_pickle(self):
        obj = TestTraitDictObject.TestClass()
        trait_dict_obj = TraitDictObject(trait=obj.trait('dict_2').trait_type, object=obj, name='a', value={})
        tdo_unpickled = pickle.loads(pickle.dumps(trait_dict_obj))
        tdo_unpickled.value_validator('1')
        tdo_unpickled.value_validator(1)
        tdo_unpickled.value_validator(True)