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
class CoerceToTempNode(CoercionNode):

    def __init__(self, arg, env):
        CoercionNode.__init__(self, arg)
        self.type = self.arg.type.as_argument_type()
        self.constant_result = self.arg.constant_result
        self.is_temp = 1
        if self.type.is_pyobject:
            self.result_ctype = py_object_type
    gil_message = 'Creating temporary Python reference'

    def analyse_types(self, env):
        return self

    def may_be_none(self):
        return self.arg.may_be_none()

    def coerce_to_boolean(self, env):
        self.arg = self.arg.coerce_to_boolean(env)
        if self.arg.is_simple():
            return self.arg
        self.type = self.arg.type
        self.result_ctype = self.type
        return self

    def generate_result_code(self, code):
        code.putln('%s = %s;' % (self.result(), self.arg.result_as(self.ctype())))
        if self.use_managed_ref:
            if not self.type.is_memoryviewslice:
                code.put_incref(self.result(), self.ctype())
            else:
                code.put_incref_memoryviewslice(self.result(), self.type, have_gil=not self.in_nogil_context)