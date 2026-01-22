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
def _get_all_specialized_permutations(fused_types, id='', f2s=()):
    fused_type, = fused_types[0].get_fused_types()
    result = []
    for newid, specific_type in enumerate(fused_type.types):
        f2s = dict(f2s)
        f2s.update({fused_type: specific_type})
        if id:
            cname = '%s_%s' % (id, newid)
        else:
            cname = str(newid)
        if len(fused_types) > 1:
            result.extend(_get_all_specialized_permutations(fused_types[1:], cname, f2s))
        else:
            result.append((cname, f2s))
    return result