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
def type_check_function(self, exact=True):
    type_name = self.name
    if type_name == 'str':
        type_check = 'PyString_Check'
    elif type_name == 'basestring':
        type_check = '__Pyx_PyBaseString_Check'
    elif type_name == 'Exception':
        type_check = '__Pyx_PyException_Check'
    elif type_name == 'bytearray':
        type_check = 'PyByteArray_Check'
    elif type_name == 'frozenset':
        type_check = 'PyFrozenSet_Check'
    elif type_name == 'int':
        type_check = '__Pyx_Py3Int_Check'
    elif type_name == 'memoryview':
        type_check = 'PyMemoryView_Check'
    else:
        type_check = 'Py%s_Check' % type_name.capitalize()
    if exact and type_name not in ('bool', 'slice', 'Exception', 'memoryview'):
        type_check += 'Exact'
    return type_check