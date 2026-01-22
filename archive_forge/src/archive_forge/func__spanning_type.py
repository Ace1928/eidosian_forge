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
def _spanning_type(type1, type2):
    if type1.is_numeric and type2.is_numeric:
        return widest_numeric_type(type1, type2)
    elif type1.is_builtin_type and type1.name == 'float' and type2.is_numeric:
        return widest_numeric_type(c_double_type, type2)
    elif type2.is_builtin_type and type2.name == 'float' and type1.is_numeric:
        return widest_numeric_type(type1, c_double_type)
    elif type1.is_extension_type and type2.is_extension_type:
        return widest_extension_type(type1, type2)
    elif type1.is_pyobject or type2.is_pyobject:
        return py_object_type
    elif type1.assignable_from(type2):
        if type1.is_extension_type and type1.typeobj_is_imported():
            return py_object_type
        return type1
    elif type2.assignable_from(type1):
        if type2.is_extension_type and type2.typeobj_is_imported():
            return py_object_type
        return type2
    elif type1.is_ptr and type2.is_ptr:
        if type1.base_type.is_cpp_class and type2.base_type.is_cpp_class:
            common_base = widest_cpp_type(type1.base_type, type2.base_type)
            if common_base:
                return CPtrType(common_base)
        return c_void_ptr_type
    else:
        return None