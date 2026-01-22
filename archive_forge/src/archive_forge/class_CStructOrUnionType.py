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
class CStructOrUnionType(CType):
    is_struct_or_union = 1
    has_attributes = 1
    exception_check = True

    def __init__(self, name, kind, scope, typedef_flag, cname, packed=False, in_cpp=False):
        self.name = name
        self.cname = cname
        self.kind = kind
        self.scope = scope
        self.typedef_flag = typedef_flag
        self.is_struct = kind == 'struct'
        self.to_py_function = '%s_to_py_%s' % (Naming.convert_func_prefix, self.specialization_name())
        self.from_py_function = '%s_from_py_%s' % (Naming.convert_func_prefix, self.specialization_name())
        self.exception_check = True
        self._convert_to_py_code = None
        self._convert_from_py_code = None
        self.packed = packed
        self.needs_cpp_construction = self.is_struct and in_cpp

    def can_coerce_to_pyobject(self, env):
        if self._convert_to_py_code is False:
            return None
        if env.outer_scope is None:
            return False
        if self._convert_to_py_code is None:
            is_union = not self.is_struct
            unsafe_union_types = set()
            safe_union_types = set()
            for member in self.scope.var_entries:
                member_type = member.type
                if not member_type.can_coerce_to_pyobject(env):
                    self.to_py_function = None
                    self._convert_to_py_code = False
                    return False
                if is_union:
                    if member_type.is_ptr or member_type.is_cpp_class:
                        unsafe_union_types.add(member_type)
                    else:
                        safe_union_types.add(member_type)
            if unsafe_union_types and (safe_union_types or len(unsafe_union_types) > 1):
                self.from_py_function = None
                self._convert_from_py_code = False
                return False
        return True

    def create_to_py_utility_code(self, env):
        if not self.can_coerce_to_pyobject(env):
            return False
        if self._convert_to_py_code is None:
            for member in self.scope.var_entries:
                member.type.create_to_py_utility_code(env)
            forward_decl = self.entry.visibility != 'extern' and (not self.typedef_flag)
            self._convert_to_py_code = ToPyStructUtilityCode(self, forward_decl, env)
        env.use_utility_code(self._convert_to_py_code)
        return True

    def can_coerce_from_pyobject(self, env):
        if env.outer_scope is None or self._convert_from_py_code is False:
            return False
        for member in self.scope.var_entries:
            if not member.type.can_coerce_from_pyobject(env):
                return False
        return True

    def create_from_py_utility_code(self, env):
        if env.outer_scope is None:
            return False
        if self._convert_from_py_code is False:
            return None
        if self._convert_from_py_code is None:
            if not self.scope.var_entries:
                return False
            for member in self.scope.var_entries:
                if not member.type.create_from_py_utility_code(env):
                    self.from_py_function = None
                    self._convert_from_py_code = False
                    return False
            context = dict(struct_type=self, var_entries=self.scope.var_entries, funcname=self.from_py_function)
            env.use_utility_code(UtilityCode.load_cached('RaiseUnexpectedTypeError', 'ObjectHandling.c'))
            from .UtilityCode import CythonUtilityCode
            self._convert_from_py_code = CythonUtilityCode.load('FromPyStructUtility' if self.is_struct else 'FromPyUnionUtility', 'CConvert.pyx', outer_module_scope=env.global_scope(), context=context)
        env.use_utility_code(self._convert_from_py_code)
        return True

    def __repr__(self):
        return '<CStructOrUnionType %s %s%s>' % (self.name, self.cname, ('', ' typedef')[self.typedef_flag])

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        if pyrex or for_display:
            base_code = self.name
        else:
            if self.typedef_flag:
                base_code = self.cname
            else:
                base_code = '%s %s' % (self.kind, self.cname)
            base_code = public_decl(base_code, dll_linkage)
        return self.base_declaration_code(base_code, entity_code)

    def __eq__(self, other):
        try:
            return isinstance(other, CStructOrUnionType) and self.name == other.name
        except AttributeError:
            return False

    def __lt__(self, other):
        try:
            return self.name < other.name
        except AttributeError:
            return False

    def __hash__(self):
        return hash(self.cname) ^ hash(self.kind)

    def is_complete(self):
        return self.scope is not None

    def attributes_known(self):
        return self.is_complete()

    def can_be_complex(self):
        fields = self.scope.var_entries
        if len(fields) != 2:
            return False
        a, b = fields
        return a.type.is_float and b.type.is_float and (a.type.empty_declaration_code() == b.type.empty_declaration_code())

    def struct_nesting_depth(self):
        child_depths = [x.type.struct_nesting_depth() for x in self.scope.var_entries]
        return max(child_depths) + 1

    def cast_code(self, expr_code):
        if self.is_struct:
            return expr_code
        return super(CStructOrUnionType, self).cast_code(expr_code)