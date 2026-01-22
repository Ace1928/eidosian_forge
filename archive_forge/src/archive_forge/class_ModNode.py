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
class ModNode(DivNode):

    def is_py_operation_types(self, type1, type2):
        return type1.is_string or type2.is_string or NumBinopNode.is_py_operation_types(self, type1, type2)

    def infer_builtin_types_operation(self, type1, type2):
        if type1 is unicode_type:
            if type2.is_builtin_type or not self.operand1.may_be_none():
                return type1
        elif type1 in (bytes_type, str_type, basestring_type):
            if type2 is unicode_type:
                return type2
            elif type2.is_numeric:
                return type1
            elif self.operand1.is_string_literal:
                if type1 is str_type or type1 is bytes_type:
                    if set(_find_formatting_types(self.operand1.value)) <= _safe_bytes_formats:
                        return type1
                return basestring_type
            elif type1 is bytes_type and (not type2.is_builtin_type):
                return None
            else:
                return basestring_type
        return None

    def zero_division_message(self):
        if self.type.is_int:
            return 'integer division or modulo by zero'
        else:
            return 'float divmod()'

    def analyse_operation(self, env):
        DivNode.analyse_operation(self, env)
        if not self.type.is_pyobject:
            if self.cdivision is None:
                self.cdivision = env.directives['cdivision'] or not self.type.signed
            if not self.cdivision and (not self.type.is_int) and (not self.type.is_float):
                error(self.pos, "mod operator not supported for type '%s'" % self.type)

    def generate_evaluation_code(self, code):
        if not self.type.is_pyobject and (not self.cdivision):
            if self.type.is_int:
                code.globalstate.use_utility_code(UtilityCode.load_cached('ModInt', 'CMath.c').specialize(self.type))
            else:
                code.globalstate.use_utility_code(UtilityCode.load_cached('ModFloat', 'CMath.c').specialize(self.type, math_h_modifier=self.type.math_h_modifier))
        NumBinopNode.generate_evaluation_code(self, code)
        self.generate_div_warning_code(code)

    def calculate_result_code(self):
        if self.cdivision:
            if self.type.is_float:
                return 'fmod%s(%s, %s)' % (self.type.math_h_modifier, self.operand1.result(), self.operand2.result())
            else:
                return '(%s %% %s)' % (self.operand1.result(), self.operand2.result())
        else:
            return '__Pyx_mod_%s(%s, %s)' % (self.type.specialization_name(), self.operand1.result(), self.operand2.result())

    def py_operation_function(self, code):
        type1, type2 = (self.operand1.type, self.operand2.type)
        if type1 is unicode_type:
            if self.operand1.may_be_none() or (type2.is_extension_type and type2.subtype_of(type1) or (type2 is py_object_type and (not isinstance(self.operand2, CoerceToPyTypeNode)))):
                return '__Pyx_PyUnicode_FormatSafe'
            else:
                return 'PyUnicode_Format'
        elif type1 is str_type:
            if self.operand1.may_be_none() or (type2.is_extension_type and type2.subtype_of(type1) or (type2 is py_object_type and (not isinstance(self.operand2, CoerceToPyTypeNode)))):
                return '__Pyx_PyString_FormatSafe'
            else:
                return '__Pyx_PyString_Format'
        return super(ModNode, self).py_operation_function(code)