import sys
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql.language import yaqltypes
def _is_specialization_of(mapping1, mapping2):
    args_mapping1, kwargs_mapping1 = mapping1
    args_mapping2, kwargs_mapping2 = mapping2
    res = False
    for a1, a2 in zip(args_mapping1, args_mapping2):
        if a2.value_type.is_specialization_of(a1.value_type):
            return False
        elif a1.value_type.is_specialization_of(a2.value_type):
            res = True
    for key, a1 in kwargs_mapping1.items():
        a2 = kwargs_mapping2[key]
        if a2.value_type.is_specialization_of(a1.value_type):
            return False
        elif a1.value_type.is_specialization_of(a2.value_type):
            res = True
    return res