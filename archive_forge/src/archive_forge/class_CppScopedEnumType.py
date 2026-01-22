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
class CppScopedEnumType(CType, EnumMixin):
    is_cpp_enum = True

    def __init__(self, name, cname, underlying_type, namespace=None, doc=None):
        self.name = name
        self.doc = doc
        self.cname = cname
        self.values = []
        self.underlying_type = underlying_type
        self.namespace = namespace

    def __str__(self):
        return self.name

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        if pyrex or for_display:
            type_name = self.name
        else:
            if self.namespace:
                type_name = '%s::%s' % (self.namespace.empty_declaration_code(), self.cname)
            else:
                type_name = '__PYX_ENUM_CLASS_DECL %s' % self.cname
            type_name = public_decl(type_name, dll_linkage)
        return self.base_declaration_code(type_name, entity_code)

    def create_from_py_utility_code(self, env):
        if self.from_py_function:
            return True
        if self.underlying_type.create_from_py_utility_code(env):
            self.from_py_function = '(%s)%s' % (self.cname, self.underlying_type.from_py_function)
        return True

    def create_to_py_utility_code(self, env):
        if self.to_py_function is not None:
            return True
        if self.entry.create_wrapper:
            self.create_enum_to_py_utility_code(env)
            return True
        if self.underlying_type.create_to_py_utility_code(env):
            self.to_py_function = '[](const %s& x){return %s((%s)x);}' % (self.cname, self.underlying_type.to_py_function, self.underlying_type.empty_declaration_code())
        return True

    def create_type_wrapper(self, env):
        from .UtilityCode import CythonUtilityCode
        rst = CythonUtilityCode.load('CppScopedEnumType', 'CpdefEnums.pyx', context={'name': self.name, 'cname': self.cname.split('::')[-1], 'items': tuple(self.values), 'underlying_type': self.underlying_type.empty_declaration_code(), 'enum_doc': self.doc, 'static_modname': env.qualified_name}, outer_module_scope=env.global_scope())
        env.use_utility_code(rst)