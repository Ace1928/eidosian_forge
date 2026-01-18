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
def widest_extension_type(type1, type2):
    if type1.typeobj_is_imported() or type2.typeobj_is_imported():
        return py_object_type
    while True:
        if type1.subtype_of(type2):
            return type2
        elif type2.subtype_of(type1):
            return type1
        type1, type2 = (type1.base_type, type2.base_type)
        if type1 is None or type2 is None:
            return py_object_type