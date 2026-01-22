import itertools
from collections import OrderedDict
from ...error import GraphQLError
from ...language import ast
from ...language.printer import print_ast
from ...pyutils.pair_set import PairSet
from ...type.definition import (GraphQLInterfaceType, GraphQLList,
from ...utils.type_comparators import is_equal_type
from ...utils.type_from_ast import type_from_ast
from .base import ValidationRule
def _same_arguments(arguments1, arguments2):
    if not (arguments1 or arguments2):
        return True
    if len(arguments1) != len(arguments2):
        return False
    arguments2_values_to_arg = {a.name.value: a for a in arguments2}
    for argument1 in arguments1:
        argument2 = arguments2_values_to_arg.get(argument1.name.value)
        if not argument2:
            return False
        if not _same_value(argument1.value, argument2.value):
            return False
    return True