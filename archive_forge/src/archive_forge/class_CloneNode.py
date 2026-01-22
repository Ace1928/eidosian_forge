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
class CloneNode(CoercionNode):
    subexprs = []
    nogil_check = None

    def __init__(self, arg):
        CoercionNode.__init__(self, arg)
        self.constant_result = arg.constant_result
        type = getattr(arg, 'type', None)
        if type:
            self.type = type
            self.result_ctype = arg.result_ctype
        arg_entry = getattr(arg, 'entry', None)
        if arg_entry:
            self.entry = arg_entry

    def result(self):
        return self.arg.result()

    def may_be_none(self):
        return self.arg.may_be_none()

    def type_dependencies(self, env):
        return self.arg.type_dependencies(env)

    def infer_type(self, env):
        return self.arg.infer_type(env)

    def analyse_types(self, env):
        self.type = self.arg.type
        self.result_ctype = self.arg.result_ctype
        self.is_temp = 1
        arg_entry = getattr(self.arg, 'entry', None)
        if arg_entry:
            self.entry = arg_entry
        return self

    def coerce_to(self, dest_type, env):
        if self.arg.is_literal:
            return self.arg.coerce_to(dest_type, env)
        return super(CloneNode, self).coerce_to(dest_type, env)

    def is_simple(self):
        return True

    def generate_evaluation_code(self, code):
        pass

    def generate_result_code(self, code):
        pass

    def generate_disposal_code(self, code):
        pass

    def generate_post_assignment_code(self, code):
        if self.is_temp:
            code.put_incref(self.result(), self.ctype())

    def free_temps(self, code):
        pass