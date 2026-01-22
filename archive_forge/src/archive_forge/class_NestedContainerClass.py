import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
class NestedContainerClass(HasTraits):
    list_of_list = List(List)
    dict_of_list = Dict(Str, List(Str))
    dict_of_union_none_or_list = Dict(Str, Union(List(), None))
    list_of_dict = List(Dict)
    dict_of_dict = Dict(Str, Dict)
    dict_of_union_none_or_dict = Dict(Str, Union(Dict(), None))
    list_of_set = List(Set)
    dict_of_set = Dict(Str, Set)
    dict_of_union_none_or_set = Dict(Str, Union(Set(), None))