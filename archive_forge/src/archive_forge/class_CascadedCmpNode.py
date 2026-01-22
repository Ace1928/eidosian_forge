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
class CascadedCmpNode(Node, CmpNode):
    child_attrs = ['operand2', 'coerced_operand2', 'cascade', 'special_bool_extra_args']
    cascade = None
    coerced_operand2 = None
    constant_result = constant_value_not_set

    def infer_type(self, env):
        return py_object_type

    def type_dependencies(self, env):
        return ()

    def has_constant_result(self):
        return self.constant_result is not constant_value_not_set and self.constant_result is not not_a_constant

    def analyse_types(self, env):
        self.operand2 = self.operand2.analyse_types(env)
        if self.cascade:
            self.cascade = self.cascade.analyse_types(env)
        return self

    def has_python_operands(self):
        return self.operand2.type.is_pyobject

    def is_cpp_comparison(self):
        return False

    def optimise_comparison(self, operand1, env, result_is_bool=False):
        if self.find_special_bool_compare_function(env, operand1, result_is_bool):
            self.is_pycmp = False
            self.type = PyrexTypes.c_bint_type
            if not operand1.type.is_pyobject:
                operand1 = operand1.coerce_to_pyobject(env)
        if self.cascade:
            operand2 = self.cascade.optimise_comparison(self.operand2, env, result_is_bool)
            if operand2 is not self.operand2:
                self.coerced_operand2 = operand2
        return operand1

    def coerce_operands_to_pyobjects(self, env):
        self.operand2 = self.operand2.coerce_to_pyobject(env)
        if self.operand2.type is dict_type and self.operator in ('in', 'not_in'):
            self.operand2 = self.operand2.as_none_safe_node("'NoneType' object is not iterable")
        if self.cascade:
            self.cascade.coerce_operands_to_pyobjects(env)

    def coerce_cascaded_operands_to_temp(self, env):
        if self.cascade:
            self.operand2 = self.operand2.coerce_to_simple(env)
            self.cascade.coerce_cascaded_operands_to_temp(env)

    def generate_evaluation_code(self, code, result, operand1, needs_evaluation=False):
        if self.type.is_pyobject:
            code.putln('if (__Pyx_PyObject_IsTrue(%s)) {' % result)
            code.put_decref(result, self.type)
        else:
            code.putln('if (%s) {' % result)
        if needs_evaluation:
            operand1.generate_evaluation_code(code)
        self.operand2.generate_evaluation_code(code)
        for extra_arg in self.special_bool_extra_args:
            extra_arg.generate_evaluation_code(code)
        self.generate_operation_code(code, result, operand1, self.operator, self.operand2)
        if self.cascade:
            self.cascade.generate_evaluation_code(code, result, self.coerced_operand2 or self.operand2, needs_evaluation=self.coerced_operand2 is not None)
        if needs_evaluation:
            operand1.generate_disposal_code(code)
            operand1.free_temps(code)
        self.operand2.generate_disposal_code(code)
        self.operand2.free_temps(code)
        code.putln('}')

    def annotate(self, code):
        self.operand2.annotate(code)
        if self.cascade:
            self.cascade.annotate(code)