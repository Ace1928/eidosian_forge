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
class StarredUnpackingNode(ExprNode):
    subexprs = ['target']
    is_starred = 1
    type = py_object_type
    is_temp = 1
    starred_expr_allowed_here = False

    def __init__(self, pos, target):
        ExprNode.__init__(self, pos, target=target)

    def analyse_declarations(self, env):
        if not self.starred_expr_allowed_here:
            error(self.pos, 'starred expression is not allowed here')
        self.target.analyse_declarations(env)

    def infer_type(self, env):
        return self.target.infer_type(env)

    def analyse_types(self, env):
        if not self.starred_expr_allowed_here:
            error(self.pos, 'starred expression is not allowed here')
        self.target = self.target.analyse_types(env)
        self.type = self.target.type
        return self

    def analyse_target_declaration(self, env):
        self.target.analyse_target_declaration(env)

    def analyse_target_types(self, env):
        self.target = self.target.analyse_target_types(env)
        self.type = self.target.type
        return self

    def calculate_result_code(self):
        return ''

    def generate_result_code(self, code):
        pass