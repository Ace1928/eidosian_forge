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
def valid_dtype(self, dtype, i=0):
    """
        Return whether type dtype can be used as the base type of a
        memoryview slice.

        We support structs, numeric types and objects
        """
    if dtype.is_complex and dtype.real_type.is_int:
        return False
    if dtype.is_struct and dtype.kind == 'struct':
        for member in dtype.scope.var_entries:
            if not self.valid_dtype(member.type):
                return False
        return True
    return dtype.is_error or (dtype.is_array and i < 8 and self.valid_dtype(dtype.base_type, i + 1)) or dtype.is_numeric or dtype.is_pyobject or dtype.is_fused or (dtype.is_typedef and self.valid_dtype(dtype.typedef_base_type))