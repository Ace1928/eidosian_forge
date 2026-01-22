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
class CompilerDirectivesNode(Node):
    """
    Sets compiler directives for the children nodes
    """
    child_attrs = ['body']

    def analyse_declarations(self, env):
        old = env.directives
        env.directives = self.directives
        self.body.analyse_declarations(env)
        env.directives = old

    def analyse_expressions(self, env):
        old = env.directives
        env.directives = self.directives
        self.body = self.body.analyse_expressions(env)
        env.directives = old
        return self

    def generate_function_definitions(self, env, code):
        env_old = env.directives
        code_old = code.globalstate.directives
        code.globalstate.directives = self.directives
        self.body.generate_function_definitions(env, code)
        env.directives = env_old
        code.globalstate.directives = code_old

    def generate_execution_code(self, code):
        old = code.globalstate.directives
        code.globalstate.directives = self.directives
        self.body.generate_execution_code(code)
        code.globalstate.directives = old

    def annotate(self, code):
        old = code.globalstate.directives
        code.globalstate.directives = self.directives
        self.body.annotate(code)
        code.globalstate.directives = old