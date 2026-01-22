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
class PythranExpr(CType):
    to_py_function = '__Pyx_pythran_to_python'
    is_pythran_expr = True
    writable = True
    has_attributes = 1

    def __init__(self, pythran_type, org_buffer=None):
        self.org_buffer = org_buffer
        self.pythran_type = pythran_type
        self.name = self.pythran_type
        self.cname = self.pythran_type
        self.from_py_function = 'from_python<%s>' % self.pythran_type
        self.scope = None

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        assert not pyrex
        return '%s %s' % (self.cname, entity_code)

    def attributes_known(self):
        if self.scope is None:
            from . import Symtab
            self.scope = scope = Symtab.CClassScope('', None, visibility='extern', parent_type=self)
            scope.directives = {}
            scope.declare_var('ndim', c_long_type, pos=None, cname='value', is_cdef=True)
            scope.declare_cproperty('shape', c_ptr_type(c_long_type), '__Pyx_PythranShapeAccessor', doc='Pythran array shape', visibility='extern', nogil=True)
        return True

    def __eq__(self, other):
        return isinstance(other, PythranExpr) and self.pythran_type == other.pythran_type

    def __ne__(self, other):
        return not (isinstance(other, PythranExpr) and self.pythran_type == other.pythran_type)

    def __hash__(self):
        return hash(self.pythran_type)