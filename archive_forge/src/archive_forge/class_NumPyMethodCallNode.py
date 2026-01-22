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
class NumPyMethodCallNode(ExprNode):
    subexprs = ['arg_tuple']
    is_temp = True
    may_return_none = True

    def generate_evaluation_code(self, code):
        code.mark_pos(self.pos)
        self.allocate_temp_result(code)
        assert self.arg_tuple.mult_factor is None
        args = self.arg_tuple.args
        for arg in args:
            arg.generate_evaluation_code(code)
        code.putln('// function evaluation code for numpy function')
        code.putln('__Pyx_call_destructor(%s);' % self.result())
        code.putln('new (&%s) decltype(%s){%s{}(%s)};' % (self.result(), self.result(), self.function_cname, ', '.join((a.pythran_result() for a in args))))