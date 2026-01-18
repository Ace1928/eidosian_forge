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
def widest_cpp_type(type1, type2):

    @cached_function
    def bases(type):
        all = set()
        for base in type.base_classes:
            all.add(base)
            all.update(bases(base))
        return all
    common_bases = bases(type1).intersection(bases(type2))
    common_bases_bases = reduce(set.union, [bases(b) for b in common_bases], set())
    candidates = [b for b in common_bases if b not in common_bases_bases]
    if len(candidates) == 1:
        return candidates[0]
    else:
        return None