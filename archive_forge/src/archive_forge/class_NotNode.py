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
class NotNode(UnopNode):
    operator = '!'
    type = PyrexTypes.c_bint_type

    def calculate_constant_result(self):
        self.constant_result = not self.operand.constant_result

    def compile_time_value(self, denv):
        operand = self.operand.compile_time_value(denv)
        try:
            return not operand
        except Exception as e:
            self.compile_time_value_error(e)

    def infer_unop_type(self, env, operand_type):
        return PyrexTypes.c_bint_type

    def analyse_types(self, env):
        self.operand = self.operand.analyse_types(env)
        operand_type = self.operand.type
        if operand_type.is_cpp_class:
            self.analyse_cpp_operation(env)
        else:
            self.operand = self.operand.coerce_to_boolean(env)
        return self

    def calculate_result_code(self):
        return '(!%s)' % self.operand.result()