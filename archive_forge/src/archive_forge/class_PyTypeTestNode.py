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
class PyTypeTestNode(CoercionNode):
    exact_builtin_type = True

    def __init__(self, arg, dst_type, env, notnone=False):
        assert dst_type.is_extension_type or dst_type.is_builtin_type, 'PyTypeTest for %s against non extension type %s' % (arg.type, dst_type)
        CoercionNode.__init__(self, arg)
        self.type = dst_type
        self.result_ctype = arg.ctype()
        self.notnone = notnone
    nogil_check = Node.gil_error
    gil_message = 'Python type test'

    def analyse_types(self, env):
        return self

    def may_be_none(self):
        if self.notnone:
            return False
        return self.arg.may_be_none()

    def is_simple(self):
        return self.arg.is_simple()

    def result_in_temp(self):
        return self.arg.result_in_temp()

    def is_ephemeral(self):
        return self.arg.is_ephemeral()

    def nonlocally_immutable(self):
        return self.arg.nonlocally_immutable()

    def reanalyse(self):
        if self.type != self.arg.type or not self.arg.is_temp:
            return self
        if not self.type.typeobj_is_available():
            return self
        if self.arg.may_be_none() and self.notnone:
            return self.arg.as_none_safe_node('Cannot convert NoneType to %.200s' % self.type.name)
        return self.arg

    def calculate_constant_result(self):
        pass

    def calculate_result_code(self):
        return self.arg.result()

    def generate_result_code(self, code):
        if self.type.typeobj_is_available():
            if self.type.is_builtin_type:
                type_test = self.type.type_test_code(self.arg.py_result(), self.notnone, exact=self.exact_builtin_type)
                code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseUnexpectedTypeError', 'ObjectHandling.c'))
            else:
                type_test = self.type.type_test_code(self.arg.py_result(), self.notnone)
                code.globalstate.use_utility_code(UtilityCode.load_cached('ExtTypeTest', 'ObjectHandling.c'))
            code.putln('if (!(%s)) %s' % (type_test, code.error_goto(self.pos)))
        else:
            error(self.pos, 'Cannot test type of extern C class without type object name specification')

    def generate_post_assignment_code(self, code):
        self.arg.generate_post_assignment_code(code)

    def allocate_temp_result(self, code):
        pass

    def release_temp_result(self, code):
        pass

    def free_temps(self, code):
        self.arg.free_temps(code)

    def free_subexpr_temps(self, code):
        self.arg.free_subexpr_temps(code)