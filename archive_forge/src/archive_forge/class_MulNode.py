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
class MulNode(NumBinopNode):
    is_sequence_mul = False

    def analyse_types(self, env):
        self.operand1 = self.operand1.analyse_types(env)
        self.operand2 = self.operand2.analyse_types(env)
        self.is_sequence_mul = self.calculate_is_sequence_mul()
        if self.is_sequence_mul:
            operand1 = self.operand1
            operand2 = self.operand2
            if operand1.is_sequence_constructor and operand1.mult_factor is None:
                return self.analyse_sequence_mul(env, operand1, operand2)
            elif operand2.is_sequence_constructor and operand2.mult_factor is None:
                return self.analyse_sequence_mul(env, operand2, operand1)
        self.analyse_operation(env)
        return self

    @staticmethod
    def is_builtin_seqmul_type(type):
        return type.is_builtin_type and type in builtin_sequence_types and (type is not memoryview_type)

    def calculate_is_sequence_mul(self):
        type1 = self.operand1.type
        type2 = self.operand2.type
        if type1 is long_type or type1.is_int:
            type1, type2 = (type2, type1)
        if type2 is long_type or type2.is_int:
            if type1.is_string or type1.is_ctuple:
                return True
            if self.is_builtin_seqmul_type(type1):
                return True
        return False

    def analyse_sequence_mul(self, env, seq, mult):
        assert seq.mult_factor is None
        seq = seq.coerce_to_pyobject(env)
        seq.mult_factor = mult
        return seq.analyse_types(env)

    def coerce_operands_to_pyobjects(self, env):
        if self.is_sequence_mul:
            if self.operand1.type.is_ctuple:
                self.operand1 = self.operand1.coerce_to_pyobject(env)
            elif self.operand2.type.is_ctuple:
                self.operand2 = self.operand2.coerce_to_pyobject(env)
            return
        super(MulNode, self).coerce_operands_to_pyobjects(env)

    def is_py_operation_types(self, type1, type2):
        return self.is_sequence_mul or super(MulNode, self).is_py_operation_types(type1, type2)

    def py_operation_function(self, code):
        if self.is_sequence_mul:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PySequenceMultiply', 'ObjectHandling.c'))
            return '__Pyx_PySequence_Multiply' if self.operand1.type.is_pyobject else '__Pyx_PySequence_Multiply_Left'
        return super(MulNode, self).py_operation_function(code)

    def infer_builtin_types_operation(self, type1, type2):
        if type1.is_builtin_type and type2.is_builtin_type:
            if self.is_builtin_seqmul_type(type1):
                return type1
            if self.is_builtin_seqmul_type(type2):
                return type2
        if type1.is_int:
            return type2
        if type2.is_int:
            return type1
        return None