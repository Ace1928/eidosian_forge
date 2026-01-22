from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
class DictComprehensionAppendNode(ComprehensionAppendNode):
    child_attrs = ['key_expr', 'value_expr']

    def analyse_expressions(self, env):
        self.key_expr = self.key_expr.analyse_expressions(env)
        if not self.key_expr.type.is_pyobject:
            self.key_expr = self.key_expr.coerce_to_pyobject(env)
        self.value_expr = self.value_expr.analyse_expressions(env)
        if not self.value_expr.type.is_pyobject:
            self.value_expr = self.value_expr.coerce_to_pyobject(env)
        return self

    def generate_execution_code(self, code):
        self.key_expr.generate_evaluation_code(code)
        self.value_expr.generate_evaluation_code(code)
        code.putln(code.error_goto_if('PyDict_SetItem(%s, (PyObject*)%s, (PyObject*)%s)' % (self.target.result(), self.key_expr.result(), self.value_expr.result()), self.pos))
        self.key_expr.generate_disposal_code(code)
        self.key_expr.free_temps(code)
        self.value_expr.generate_disposal_code(code)
        self.value_expr.free_temps(code)

    def generate_function_definitions(self, env, code):
        self.key_expr.generate_function_definitions(env, code)
        self.value_expr.generate_function_definitions(env, code)

    def annotate(self, code):
        self.key_expr.annotate(code)
        self.value_expr.annotate(code)