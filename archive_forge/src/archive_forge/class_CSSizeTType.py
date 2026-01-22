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
class CSSizeTType(CIntType):
    to_py_function = 'PyInt_FromSsize_t'
    from_py_function = 'PyInt_AsSsize_t'

    def sign_and_name(self):
        return 'Py_ssize_t'