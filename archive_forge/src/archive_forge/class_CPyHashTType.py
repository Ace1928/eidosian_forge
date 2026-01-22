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
class CPyHashTType(CIntType):
    to_py_function = '__Pyx_PyInt_FromHash_t'
    from_py_function = '__Pyx_PyInt_AsHash_t'

    def sign_and_name(self):
        return 'Py_hash_t'