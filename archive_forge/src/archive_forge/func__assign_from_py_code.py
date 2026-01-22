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
def _assign_from_py_code(self, source_code, result_code, error_pos, code, from_py_function=None, error_condition=None, extra_args=None, special_none_cvalue=None):
    args = ', ' + ', '.join(('%s' % arg for arg in extra_args)) if extra_args else ''
    convert_call = '%s(%s%s)' % (from_py_function or self.from_py_function, source_code, args)
    if self.is_enum:
        convert_call = typecast(self, c_long_type, convert_call)
    if special_none_cvalue:
        convert_call = '(__Pyx_Py_IsNone(%s) ? (%s) : (%s))' % (source_code, special_none_cvalue, convert_call)
    return '%s = %s; %s' % (result_code, convert_call, code.error_goto_if(error_condition or self.error_condition(result_code), error_pos))