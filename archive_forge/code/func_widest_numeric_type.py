from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
def widest_numeric_type(type1, type2):
    """Given two numeric types, return the narrowest type encompassing both of them.
    """
    if type1.is_reference:
        type1 = type1.ref_base_type
    if type2.is_reference:
        type2 = type2.ref_base_type
    if type1.is_cv_qualified:
        type1 = type1.cv_base_type
    if type2.is_cv_qualified:
        type2 = type2.cv_base_type
    if type1 == type2:
        widest_type = type1
    elif type1.is_complex or type2.is_complex:

        def real_type(ntype):
            if ntype.is_complex:
                return ntype.real_type
            return ntype
        widest_type = CComplexType(widest_numeric_type(real_type(type1), real_type(type2)))
        if type1 is soft_complex_type or type2 is soft_complex_type:
            type1_is_other_complex = type1 is not soft_complex_type and type1.is_complex
            type2_is_other_complex = type2 is not soft_complex_type and type2.is_complex
            if not type1_is_other_complex and (not type2_is_other_complex) and (widest_type.real_type == soft_complex_type.real_type):
                widest_type = soft_complex_type
    elif type1.is_enum and type2.is_enum:
        widest_type = c_int_type
    elif type1.rank < type2.rank:
        widest_type = type2
    elif type1.rank > type2.rank:
        widest_type = type1
    elif type1.signed < type2.signed:
        widest_type = type1
    elif type1.signed > type2.signed:
        widest_type = type2
    elif type1.is_typedef > type2.is_typedef:
        widest_type = type1
    else:
        widest_type = type2
    return widest_type