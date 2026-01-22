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
class CImportStatNode(StatNode):
    child_attrs = []
    is_absolute = False

    def analyse_declarations(self, env):
        if not env.is_module_scope:
            error(self.pos, 'cimport only allowed at module level')
            return
        module_scope = env.find_module(self.module_name, self.pos, relative_level=0 if self.is_absolute else -1)
        if '.' in self.module_name:
            names = [EncodedString(name) for name in self.module_name.split('.')]
            top_name = names[0]
            top_module_scope = env.context.find_submodule(top_name)
            module_scope = top_module_scope
            for name in names[1:]:
                submodule_scope = module_scope.find_submodule(name)
                module_scope.declare_module(name, submodule_scope, self.pos)
                module_scope = submodule_scope
            if self.as_name:
                env.declare_module(self.as_name, module_scope, self.pos)
            else:
                env.add_imported_module(module_scope)
                env.declare_module(top_name, top_module_scope, self.pos)
        else:
            name = self.as_name or self.module_name
            entry = env.declare_module(name, module_scope, self.pos)
            entry.known_standard_library_import = self.module_name
        if self.module_name in utility_code_for_cimports:
            env.use_utility_code(utility_code_for_cimports[self.module_name]())

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        if self.module_name == 'numpy':
            cimport_numpy_check(self, code)