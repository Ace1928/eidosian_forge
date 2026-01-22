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
class AsyncIteratorNode(ScopedExprNode):
    subexprs = ['sequence']
    is_async = True
    type = py_object_type
    is_temp = 1
    has_local_scope = False

    def infer_type(self, env):
        return py_object_type

    def analyse_types(self, env):
        if self.expr_scope:
            env = self.expr_scope
        self.sequence = self.sequence.analyse_types(env)
        if not self.sequence.type.is_pyobject:
            error(self.pos, 'async for loops not allowed on C/C++ types')
            self.sequence = self.sequence.coerce_to_pyobject(env)
        return self

    def generate_result_code(self, code):
        code.globalstate.use_utility_code(UtilityCode.load_cached('AsyncIter', 'Coroutine.c'))
        code.putln('%s = __Pyx_Coroutine_GetAsyncIter(%s); %s' % (self.result(), self.sequence.py_result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)