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
class NewExprNode(AtomicExprNode):
    type = None

    def infer_type(self, env):
        type = self.cppclass.analyse_as_type(env)
        if type is None or not type.is_cpp_class:
            error(self.pos, 'new operator can only be applied to a C++ class')
            self.type = error_type
            return
        self.cpp_check(env)
        constructor = type.get_constructor(self.pos)
        self.class_type = type
        self.entry = constructor
        self.type = constructor.type
        return self.type

    def analyse_types(self, env):
        if self.type is None:
            self.infer_type(env)
        return self

    def may_be_none(self):
        return False

    def generate_result_code(self, code):
        pass

    def calculate_result_code(self):
        return 'new ' + self.class_type.empty_declaration_code()