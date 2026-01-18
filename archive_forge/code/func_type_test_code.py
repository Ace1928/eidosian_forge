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
def type_test_code(self, py_arg, notnone=False):
    none_check = '((%s) == Py_None)' % py_arg
    type_check = 'likely(__Pyx_TypeTest(%s, %s))' % (py_arg, self.typeptr_cname)
    if notnone:
        return type_check
    else:
        return 'likely(%s || %s)' % (none_check, type_check)