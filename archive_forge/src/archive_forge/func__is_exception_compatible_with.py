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
def _is_exception_compatible_with(self, other_type):
    if self.exception_check == '+' and other_type.exception_check != '+':
        return 0
    if not other_type.exception_check or other_type.exception_value is not None:
        if other_type.exception_check and (not (self.exception_check or self.exception_value)):
            return 1
        if not self._same_exception_value(other_type.exception_value):
            return 0
        if self.exception_check and self.exception_check != other_type.exception_check:
            return 0
    return 1