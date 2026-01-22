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
class BlockNode(object):

    def generate_cached_builtins_decls(self, env, code):
        entries = env.global_scope().undeclared_cached_builtins
        for entry in entries:
            code.globalstate.add_cached_builtin_decl(entry)
        del entries[:]

    def generate_lambda_definitions(self, env, code):
        for node in env.lambda_defs:
            node.generate_function_definitions(env, code)