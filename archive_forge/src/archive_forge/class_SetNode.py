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
class SetNode(ExprNode):
    """
    Set constructor.
    """
    subexprs = ['args']
    type = set_type
    is_set_literal = True
    gil_message = 'Constructing Python set'

    def analyse_types(self, env):
        for i in range(len(self.args)):
            arg = self.args[i]
            arg = arg.analyse_types(env)
            self.args[i] = arg.coerce_to_pyobject(env)
        self.type = set_type
        self.is_temp = 1
        return self

    def may_be_none(self):
        return False

    def calculate_constant_result(self):
        self.constant_result = {arg.constant_result for arg in self.args}

    def compile_time_value(self, denv):
        values = [arg.compile_time_value(denv) for arg in self.args]
        try:
            return set(values)
        except Exception as e:
            self.compile_time_value_error(e)

    def generate_evaluation_code(self, code):
        for arg in self.args:
            arg.generate_evaluation_code(code)
        self.allocate_temp_result(code)
        code.putln('%s = PySet_New(0); %s' % (self.result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)
        for arg in self.args:
            code.put_error_if_neg(self.pos, 'PySet_Add(%s, %s)' % (self.result(), arg.py_result()))
            arg.generate_disposal_code(code)
            arg.free_temps(code)