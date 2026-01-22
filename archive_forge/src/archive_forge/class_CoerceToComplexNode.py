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
class CoerceToComplexNode(CoercionNode):

    def __init__(self, arg, dst_type, env):
        if arg.type.is_complex:
            arg = arg.coerce_to_simple(env)
        self.type = dst_type
        CoercionNode.__init__(self, arg)
        dst_type.create_declaration_utility_code(env)

    def calculate_result_code(self):
        if self.arg.type.is_complex:
            real_part = self.arg.type.real_code(self.arg.result())
            imag_part = self.arg.type.imag_code(self.arg.result())
        else:
            real_part = self.arg.result()
            imag_part = '0'
        return '%s(%s, %s)' % (self.type.from_parts, real_part, imag_part)

    def generate_result_code(self, code):
        pass

    def analyse_types(self, env):
        return self