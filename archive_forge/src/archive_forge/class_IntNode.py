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
class IntNode(ConstNode):
    unsigned = ''
    longness = ''
    is_c_literal = None

    @property
    def hex_value(self):
        return Utils.strip_py2_long_suffix(hex(Utils.str_to_number(self.value)))

    @property
    def base_10_value(self):
        return str(Utils.str_to_number(self.value))

    def __init__(self, pos, **kwds):
        ExprNode.__init__(self, pos, **kwds)
        if 'type' not in kwds:
            self.type = self.find_suitable_type_for_value()

    def find_suitable_type_for_value(self):
        if self.constant_result is constant_value_not_set:
            try:
                self.calculate_constant_result()
            except ValueError:
                pass
        if self.is_c_literal or not self.has_constant_result() or self.unsigned or (self.longness == 'LL'):
            rank = self.longness == 'LL' and 2 or 1
            suitable_type = PyrexTypes.modifiers_and_name_to_type[not self.unsigned, rank, 'int']
            if self.type:
                suitable_type = PyrexTypes.widest_numeric_type(suitable_type, self.type)
        elif -2 ** 31 <= self.constant_result < 2 ** 31:
            if self.type and self.type.is_int:
                suitable_type = self.type
            else:
                suitable_type = PyrexTypes.c_long_type
        else:
            suitable_type = PyrexTypes.py_object_type
        return suitable_type

    def coerce_to(self, dst_type, env):
        if self.type is dst_type:
            return self
        elif dst_type.is_float:
            if self.has_constant_result():
                return FloatNode(self.pos, value='%d.0' % int(self.constant_result), type=dst_type, constant_result=float(self.constant_result))
            else:
                return FloatNode(self.pos, value=self.value, type=dst_type, constant_result=not_a_constant)
        if dst_type.is_numeric and (not dst_type.is_complex):
            node = IntNode(self.pos, value=self.value, constant_result=self.constant_result, type=dst_type, is_c_literal=True, unsigned=self.unsigned, longness=self.longness)
            return node
        elif dst_type.is_pyobject:
            node = IntNode(self.pos, value=self.value, constant_result=self.constant_result, type=PyrexTypes.py_object_type, is_c_literal=False, unsigned=self.unsigned, longness=self.longness)
        else:
            node = IntNode(self.pos, value=self.value, constant_result=self.constant_result, unsigned=self.unsigned, longness=self.longness)
        return ConstNode.coerce_to(node, dst_type, env)

    def coerce_to_boolean(self, env):
        return IntNode(self.pos, value=self.value, constant_result=self.constant_result, type=PyrexTypes.c_bint_type, unsigned=self.unsigned, longness=self.longness)

    def generate_evaluation_code(self, code):
        if self.type.is_pyobject:
            value = Utils.str_to_number(self.value)
            formatter = hex if value > 10 ** 13 else str
            plain_integer_string = formatter(value)
            plain_integer_string = Utils.strip_py2_long_suffix(plain_integer_string)
            self.result_code = code.get_py_int(plain_integer_string, self.longness)
        else:
            self.result_code = self.get_constant_c_result_code()

    def get_constant_c_result_code(self):
        unsigned, longness = (self.unsigned, self.longness)
        literal = self.value_as_c_integer_string()
        if not (unsigned or longness) and self.type.is_int and (literal[0] == '-') and (literal[1] != '0'):
            if self.type.rank >= PyrexTypes.c_longlong_type.rank:
                longness = 'LL'
            elif self.type.rank >= PyrexTypes.c_long_type.rank:
                longness = 'L'
        return literal + unsigned + longness

    def value_as_c_integer_string(self):
        value = self.value
        if len(value) <= 2:
            return value
        neg_sign = ''
        if value[0] == '-':
            neg_sign = '-'
            value = value[1:]
        if value[0] == '0':
            literal_type = value[1]
            if neg_sign and literal_type in 'oOxX0123456789' and value[2:].isdigit():
                value = str(Utils.str_to_number(value))
            elif literal_type in 'oO':
                value = '0' + value[2:]
            elif literal_type in 'bB':
                value = str(int(value[2:], 2))
        elif value.isdigit() and (not self.unsigned) and (not self.longness):
            if not neg_sign:
                value = '0x%X' % int(value)
        return neg_sign + value

    def calculate_result_code(self):
        return self.result_code

    def calculate_constant_result(self):
        self.constant_result = Utils.str_to_number(self.value)

    def compile_time_value(self, denv):
        return Utils.str_to_number(self.value)