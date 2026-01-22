from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
class BuiltinScope(Scope):
    is_builtin_scope = True

    def __init__(self):
        if Options.pre_import is None:
            Scope.__init__(self, '__builtin__', None, None)
        else:
            Scope.__init__(self, '__builtin__', PreImportScope(), None)
        self.type_names = {}
        self.declare_var('bool', py_object_type, None, '((PyObject*)&PyBool_Type)')

    def lookup(self, name, language_level=None, str_is_str=None):
        if name == 'str':
            if str_is_str is None:
                str_is_str = language_level in (None, 2)
            if not str_is_str:
                name = 'unicode'
        return Scope.lookup(self, name)

    def declare_builtin(self, name, pos):
        if not hasattr(builtins, name):
            if self.outer_scope is not None:
                return self.outer_scope.declare_builtin(name, pos)
            elif Options.error_on_unknown_names:
                error(pos, 'undeclared name not builtin: %s' % name)
            else:
                warning(pos, 'undeclared name not builtin: %s' % name, 2)

    def declare_builtin_cfunction(self, name, type, cname, python_equiv=None, utility_code=None):
        name = EncodedString(name)
        entry = self.declare_cfunction(name, type, None, cname, visibility='extern', utility_code=utility_code)
        if python_equiv:
            if python_equiv == '*':
                python_equiv = name
            else:
                python_equiv = EncodedString(python_equiv)
            var_entry = Entry(python_equiv, python_equiv, py_object_type)
            var_entry.qualified_name = self.qualify_name(name)
            var_entry.is_variable = 1
            var_entry.is_builtin = 1
            var_entry.utility_code = utility_code
            var_entry.scope = entry.scope
            entry.as_variable = var_entry
        return entry

    def declare_builtin_type(self, name, cname, utility_code=None, objstruct_cname=None, type_class=PyrexTypes.BuiltinObjectType):
        name = EncodedString(name)
        type = type_class(name, cname, objstruct_cname)
        scope = CClassScope(name, outer_scope=None, visibility='extern', parent_type=type)
        scope.directives = {}
        if name == 'bool':
            type.is_final_type = True
        type.set_scope(scope)
        self.type_names[name] = 1
        entry = self.declare_type(name, type, None, visibility='extern')
        entry.utility_code = utility_code
        var_entry = Entry(name=entry.name, type=self.lookup('type').type, pos=entry.pos, cname=entry.type.typeptr_cname)
        var_entry.qualified_name = self.qualify_name(name)
        var_entry.is_variable = 1
        var_entry.is_cglobal = 1
        var_entry.is_readonly = 1
        var_entry.is_builtin = 1
        var_entry.utility_code = utility_code
        var_entry.scope = self
        if Options.cache_builtins:
            var_entry.is_const = True
        entry.as_variable = var_entry
        return type

    def builtin_scope(self):
        return self