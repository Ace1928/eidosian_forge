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
class FusedType(CType):
    """
    Represents a Fused Type. All it needs to do is keep track of the types
    it aggregates, as it will be replaced with its specific version wherever
    needed.

    See http://wiki.cython.org/enhancements/fusedtypes

    types           [PyrexType]             is the list of types to be fused
    name            str                     the name of the ctypedef
    """
    is_fused = 1
    exception_check = 0

    def __init__(self, types, name=None):
        flattened_types = []
        for t in types:
            if t.is_fused:
                if isinstance(t, FusedType):
                    t_types = t.types
                else:
                    t_fused_types = t.get_fused_types()
                    t_types = []
                    for substitution in product(*[fused_type.types for fused_type in t_fused_types]):
                        t_types.append(t.specialize({fused_type: sub for fused_type, sub in zip(t_fused_types, substitution)}))
                for subtype in t_types:
                    if subtype not in flattened_types:
                        flattened_types.append(subtype)
            elif t not in flattened_types:
                flattened_types.append(t)
        self.types = flattened_types
        self.name = name

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        if pyrex or for_display:
            return self.name
        raise Exception('This may never happen, please report a bug')

    def __repr__(self):
        return 'FusedType(name=%r)' % self.name

    def specialize(self, values):
        if self in values:
            return values[self]
        else:
            raise CannotSpecialize()

    def get_fused_types(self, result=None, seen=None, include_function_return_type=False):
        if result is None:
            return [self]
        if self not in seen:
            result.append(self)
            seen.add(self)