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
def validate_memslice_dtype(self, pos):
    if not self.valid_dtype(self.dtype):
        error(pos, 'Invalid base type for memoryview slice: %s' % self.dtype)