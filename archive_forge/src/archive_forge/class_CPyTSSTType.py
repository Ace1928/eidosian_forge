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
class CPyTSSTType(CType):
    declaration_value = 'Py_tss_NEEDS_INIT'

    def __repr__(self):
        return '<Py_tss_t>'

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        if pyrex or for_display:
            base_code = 'Py_tss_t'
        else:
            base_code = public_decl('Py_tss_t', dll_linkage)
        return self.base_declaration_code(base_code, entity_code)