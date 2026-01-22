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
class CAnonEnumType(CIntType):
    is_enum = 1

    def sign_and_name(self):
        return 'int'

    def specialization_name(self):
        return '__pyx_anon_enum'