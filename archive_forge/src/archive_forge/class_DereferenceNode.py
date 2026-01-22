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
class DereferenceNode(CUnopNode):
    operator = '*'

    def infer_unop_type(self, env, operand_type):
        if operand_type.is_ptr:
            return operand_type.base_type
        else:
            return PyrexTypes.error_type

    def analyse_c_operation(self, env):
        if self.operand.type.is_ptr:
            if env.is_cpp:
                self.type = PyrexTypes.CReferenceType(self.operand.type.base_type)
            else:
                self.type = self.operand.type.base_type
        else:
            self.type_error()

    def calculate_result_code(self):
        return '(*%s)' % self.operand.result()