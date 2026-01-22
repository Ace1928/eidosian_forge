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
class ProxyNode(CoercionNode):
    """
    A node that should not be replaced by transforms or other means,
    and hence can be useful to wrap the argument to a clone node

    MyNode    -> ProxyNode -> ArgNode
    CloneNode -^
    """
    nogil_check = None

    def __init__(self, arg):
        super(ProxyNode, self).__init__(arg)
        self.constant_result = arg.constant_result
        self.update_type_and_entry()

    def analyse_types(self, env):
        self.arg = self.arg.analyse_expressions(env)
        self.update_type_and_entry()
        return self

    def infer_type(self, env):
        return self.arg.infer_type(env)

    def update_type_and_entry(self):
        type = getattr(self.arg, 'type', None)
        if type:
            self.type = type
            self.result_ctype = self.arg.result_ctype
        arg_entry = getattr(self.arg, 'entry', None)
        if arg_entry:
            self.entry = arg_entry

    def generate_result_code(self, code):
        self.arg.generate_result_code(code)

    def result(self):
        return self.arg.result()

    def is_simple(self):
        return self.arg.is_simple()

    def may_be_none(self):
        return self.arg.may_be_none()

    def generate_evaluation_code(self, code):
        self.arg.generate_evaluation_code(code)

    def generate_disposal_code(self, code):
        self.arg.generate_disposal_code(code)

    def free_temps(self, code):
        self.arg.free_temps(code)