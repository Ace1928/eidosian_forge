import copy
import pickle
import sys
import unittest
from unittest import mock
from traits.api import HasTraits
from traits.trait_dict_object import TraitDict, TraitDictEvent, TraitDictObject
from traits.trait_errors import TraitError
from traits.trait_types import Dict, Int, Str
 Test string representation of a subclass of the TraitDictEvent
        class. 