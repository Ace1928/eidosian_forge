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
class StringNode(PyConstNode):
    type = str_type
    is_string_literal = True
    is_identifier = None
    unicode_value = None

    def calculate_constant_result(self):
        if self.unicode_value is not None:
            self.constant_result = self.unicode_value

    def analyse_as_type(self, env):
        return _analyse_name_as_type(self.unicode_value or self.value.decode('ISO8859-1'), self.pos, env)

    def as_sliced_node(self, start, stop, step=None):
        value = type(self.value)(self.value[start:stop:step])
        value.encoding = self.value.encoding
        if self.unicode_value is not None:
            if StringEncoding.string_contains_surrogates(self.unicode_value[:stop]):
                return None
            unicode_value = StringEncoding.EncodedString(self.unicode_value[start:stop:step])
        else:
            unicode_value = None
        return StringNode(self.pos, value=value, unicode_value=unicode_value, constant_result=value, is_identifier=self.is_identifier)

    def coerce_to(self, dst_type, env):
        if dst_type is not py_object_type and (not str_type.subtype_of(dst_type)):
            if not dst_type.is_pyobject:
                return BytesNode(self.pos, value=self.value).coerce_to(dst_type, env)
            if dst_type is not Builtin.basestring_type:
                self.check_for_coercion_error(dst_type, env, fail=True)
        return self

    def can_coerce_to_char_literal(self):
        return not self.is_identifier and len(self.value) == 1

    def generate_evaluation_code(self, code):
        self.result_code = code.get_py_string_const(self.value, identifier=self.is_identifier, is_str=True, unicode_value=self.unicode_value)

    def get_constant_c_result_code(self):
        return None

    def calculate_result_code(self):
        return self.result_code

    def compile_time_value(self, env):
        if self.value.is_unicode:
            return self.value
        if not IS_PYTHON3:
            return self.value.byteencode()
        if self.unicode_value is not None:
            return self.unicode_value
        return self.value.decode('iso8859-1')