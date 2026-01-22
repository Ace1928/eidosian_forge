from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
class CEnumDefNode(StatNode):
    child_attrs = ['items', 'underlying_type']
    doc = None

    def declare(self, env):
        doc = None
        if Options.docstrings:
            doc = embed_position(self.pos, self.doc)
        self.entry = env.declare_enum(self.name, self.pos, cname=self.cname, scoped=self.scoped, typedef_flag=self.typedef_flag, visibility=self.visibility, api=self.api, create_wrapper=self.create_wrapper, doc=doc)

    def analyse_declarations(self, env):
        scope = None
        underlying_type = self.underlying_type.analyse(env)
        if not underlying_type.is_int:
            error(self.underlying_type.pos, 'underlying type is not an integral type')
        self.entry.type.underlying_type = underlying_type
        if self.scoped and self.items is not None:
            scope = CppScopedEnumScope(self.name, env)
            scope.type = self.entry.type
            scope.directives = env.directives
        else:
            scope = env
        if self.items is not None:
            if self.in_pxd and (not env.in_cinclude):
                self.entry.defined_in_pxd = 1
            is_declared_enum = self.visibility != 'extern'
            next_int_enum_value = 0 if is_declared_enum else None
            for item in self.items:
                item.analyse_enum_declarations(scope, self.entry, next_int_enum_value)
                if is_declared_enum:
                    next_int_enum_value = 1 + (item.entry.enum_int_value if item.entry.enum_int_value is not None else next_int_enum_value)

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        if self.scoped:
            return
        if self.visibility == 'public' or self.api:
            code.mark_pos(self.pos)
            temp = code.funcstate.allocate_temp(PyrexTypes.py_object_type, manage_ref=True)
            for item in self.entry.enum_values:
                code.putln('%s = PyInt_FromLong(%s); %s' % (temp, item.cname, code.error_goto_if_null(temp, item.pos)))
                code.put_gotref(temp, PyrexTypes.py_object_type)
                code.putln('if (PyDict_SetItemString(%s, "%s", %s) < 0) %s' % (Naming.moddict_cname, item.name, temp, code.error_goto(item.pos)))
                code.put_decref_clear(temp, PyrexTypes.py_object_type)
            code.funcstate.release_temp(temp)