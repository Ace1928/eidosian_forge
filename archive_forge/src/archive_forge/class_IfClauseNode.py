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
class IfClauseNode(Node):
    child_attrs = ['condition', 'body']
    branch_hint = None

    def analyse_declarations(self, env):
        self.body.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.condition = self.condition.analyse_temp_boolean_expression(env)
        self.body = self.body.analyse_expressions(env)
        return self

    def generate_execution_code(self, code, end_label, is_last):
        self.condition.generate_evaluation_code(code)
        code.mark_pos(self.pos)
        condition = self.condition.result()
        if self.branch_hint:
            condition = '%s(%s)' % (self.branch_hint, condition)
        code.putln('if (%s) {' % condition)
        self.condition.generate_disposal_code(code)
        self.condition.free_temps(code)
        self.body.generate_execution_code(code)
        code.mark_pos(self.pos, trace=False)
        if not (is_last or self.body.is_terminator):
            code.put_goto(end_label)
        code.putln('}')

    def generate_function_definitions(self, env, code):
        self.condition.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)

    def annotate(self, code):
        self.condition.annotate(code)
        self.body.annotate(code)