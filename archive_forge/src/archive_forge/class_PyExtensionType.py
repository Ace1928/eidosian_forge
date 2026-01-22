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
class PyExtensionType(PyObjectType):
    is_extension_type = 1
    has_attributes = 1
    early_init = 1
    objtypedef_cname = None
    dataclass_fields = None
    multiple_bases = False
    has_sequence_flag = False

    def __init__(self, name, typedef_flag, base_type, is_external=0, check_size=None):
        self.name = name
        self.scope = None
        self.typedef_flag = typedef_flag
        if base_type is not None:
            base_type.is_subclassed = True
        self.base_type = base_type
        self.module_name = None
        self.objstruct_cname = None
        self.typeobj_cname = None
        self.typeptr_cname = None
        self.vtabslot_cname = None
        self.vtabstruct_cname = None
        self.vtabptr_cname = None
        self.vtable_cname = None
        self.is_external = is_external
        self.check_size = check_size or 'warn'
        self.defered_declarations = []

    def set_scope(self, scope):
        self.scope = scope
        if scope:
            scope.parent_type = self

    def needs_nonecheck(self):
        return True

    def subtype_of_resolved_type(self, other_type):
        if other_type.is_extension_type or other_type.is_builtin_type:
            return self is other_type or (self.base_type and self.base_type.subtype_of(other_type))
        else:
            return other_type is py_object_type

    def typeobj_is_available(self):
        return self.typeptr_cname

    def typeobj_is_imported(self):
        return self.typeobj_cname is None and self.module_name is not None

    def assignable_from(self, src_type):
        if self == src_type:
            return True
        if isinstance(src_type, PyExtensionType):
            if src_type.base_type is not None:
                return self.assignable_from(src_type.base_type)
        if isinstance(src_type, BuiltinObjectType):
            return self.module_name == '__builtin__' and self.name == src_type.name
        return False

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0, deref=0):
        if pyrex or for_display:
            base_code = self.name
        else:
            if self.typedef_flag:
                objstruct = self.objstruct_cname
            else:
                objstruct = 'struct %s' % self.objstruct_cname
            base_code = public_decl(objstruct, dll_linkage)
            if deref:
                assert not entity_code
            else:
                entity_code = '*%s' % entity_code
        return self.base_declaration_code(base_code, entity_code)

    def type_test_code(self, py_arg, notnone=False):
        none_check = '((%s) == Py_None)' % py_arg
        type_check = 'likely(__Pyx_TypeTest(%s, %s))' % (py_arg, self.typeptr_cname)
        if notnone:
            return type_check
        else:
            return 'likely(%s || %s)' % (none_check, type_check)

    def attributes_known(self):
        return self.scope is not None

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<PyExtensionType %s%s>' % (self.scope.class_name, ('', ' typedef')[self.typedef_flag])

    def py_type_name(self):
        if not self.module_name:
            return self.name
        return "__import__(%r, None, None, ['']).%s" % (self.module_name, self.name)