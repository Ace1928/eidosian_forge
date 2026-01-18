import copy
import pickle
import sys
import unittest
from unittest import mock
from traits.api import HasTraits
from traits.trait_dict_object import TraitDict, TraitDictEvent, TraitDictObject
from traits.trait_errors import TraitError
from traits.trait_types import Dict, Int, Str
def test_trait_dict_object_pickle(self):
    obj = TestTraitDictObject.TestClass()
    trait_dict_obj = TraitDictObject(trait=obj.trait('dict_2').trait_type, object=obj, name='a', value={})
    tdo_unpickled = pickle.loads(pickle.dumps(trait_dict_obj))
    tdo_unpickled.value_validator('1')
    tdo_unpickled.value_validator(1)
    tdo_unpickled.value_validator(True)