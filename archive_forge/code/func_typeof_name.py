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
def typeof_name(self):
    """
        Return the string with which fused python functions can be indexed.
        """
    if self.is_builtin_type or self.py_type_name() == 'object':
        index_name = self.py_type_name()
    else:
        index_name = str(self)
    return index_name