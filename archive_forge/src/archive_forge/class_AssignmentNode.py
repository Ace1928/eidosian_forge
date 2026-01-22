from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
class AssignmentNode(StatNode):

    def _warn_on_const_assignment(self, lhs, rhs):
        rhs_t = rhs.type
        lhs_t = lhs.type
        if rhs_t.is_ptr and rhs_t.base_type.is_const and lhs_t.is_ptr and (not lhs_t.base_type.is_const):
            warning(self.pos, "Assigning to '{}' from '{}' discards const qualifier".format(lhs_t, rhs_t), level=1)

    def _check_const_assignment(self, node):
        if isinstance(node, AssignmentNode):
            self._warn_on_const_assignment(node.lhs, node.rhs)

    def analyse_expressions(self, env):
        node = self.analyse_types(env)
        self._check_const_assignment(node)
        if isinstance(node, AssignmentNode) and (not isinstance(node, ParallelAssignmentNode)):
            if node.rhs.type.is_ptr and node.rhs.is_ephemeral():
                error(self.pos, 'Storing unsafe C derivative of temporary Python reference')
        return node

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        self.generate_rhs_evaluation_code(code)
        self.generate_assignment_code(code)