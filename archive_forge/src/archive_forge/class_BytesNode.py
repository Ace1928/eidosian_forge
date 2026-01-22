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
class BytesNode(ConstNode):
    is_string_literal = True
    type = bytes_type

    def calculate_constant_result(self):
        self.constant_result = self.value

    def as_sliced_node(self, start, stop, step=None):
        value = StringEncoding.bytes_literal(self.value[start:stop:step], self.value.encoding)
        return BytesNode(self.pos, value=value, constant_result=value)

    def compile_time_value(self, denv):
        return self.value.byteencode()

    def analyse_as_type(self, env):
        return _analyse_name_as_type(self.value.decode('ISO8859-1'), self.pos, env)

    def can_coerce_to_char_literal(self):
        return len(self.value) == 1

    def coerce_to_boolean(self, env):
        bool_value = bool(self.value)
        return BoolNode(self.pos, value=bool_value, constant_result=bool_value)

    def coerce_to(self, dst_type, env):
        if self.type == dst_type:
            return self
        if dst_type.is_int:
            if not self.can_coerce_to_char_literal():
                error(self.pos, 'Only single-character string literals can be coerced into ints.')
                return self
            if dst_type.is_unicode_char:
                error(self.pos, 'Bytes literals cannot coerce to Py_UNICODE/Py_UCS4, use a unicode literal instead.')
                return self
            return CharNode(self.pos, value=self.value, constant_result=ord(self.value))
        node = BytesNode(self.pos, value=self.value, constant_result=self.constant_result)
        if dst_type.is_pyobject:
            if dst_type in (py_object_type, Builtin.bytes_type):
                node.type = Builtin.bytes_type
            else:
                self.check_for_coercion_error(dst_type, env, fail=True)
            return node
        elif dst_type in (PyrexTypes.c_char_ptr_type, PyrexTypes.c_const_char_ptr_type):
            node.type = dst_type
            return node
        elif dst_type in (PyrexTypes.c_uchar_ptr_type, PyrexTypes.c_const_uchar_ptr_type, PyrexTypes.c_void_ptr_type):
            node.type = PyrexTypes.c_const_char_ptr_type if dst_type == PyrexTypes.c_const_uchar_ptr_type else PyrexTypes.c_char_ptr_type
            return CastNode(node, dst_type)
        elif dst_type.assignable_from(PyrexTypes.c_char_ptr_type):
            if not dst_type.is_cpp_class or dst_type.is_const:
                node.type = dst_type
                return node
        return ConstNode.coerce_to(node, dst_type, env)

    def generate_evaluation_code(self, code):
        if self.type.is_pyobject:
            result = code.get_py_string_const(self.value)
        elif self.type.is_const:
            result = code.get_string_const(self.value)
        else:
            literal = self.value.as_c_string_literal()
            result = typecast(self.type, PyrexTypes.c_void_ptr_type, literal)
        self.result_code = result

    def get_constant_c_result_code(self):
        return None

    def calculate_result_code(self):
        return self.result_code