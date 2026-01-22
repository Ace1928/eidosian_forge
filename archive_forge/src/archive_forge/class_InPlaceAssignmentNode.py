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
class InPlaceAssignmentNode(AssignmentNode):
    child_attrs = ['lhs', 'rhs']

    def analyse_declarations(self, env):
        self.lhs.analyse_target_declaration(env)

    def analyse_types(self, env):
        self.rhs = self.rhs.analyse_types(env)
        self.lhs = self.lhs.analyse_target_types(env)
        if self.lhs.is_memview_index or self.lhs.is_buffer_access:
            self.rhs = self.rhs.coerce_to(self.lhs.type, env)
        elif self.lhs.type.is_string and self.operator in '+-':
            self.rhs = self.rhs.coerce_to(PyrexTypes.c_py_ssize_t_type, env)
        return self

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        lhs, rhs = (self.lhs, self.rhs)
        rhs.generate_evaluation_code(code)
        lhs.generate_subexpr_evaluation_code(code)
        c_op = self.operator
        if c_op == '//':
            c_op = '/'
        elif c_op == '**':
            error(self.pos, 'No C inplace power operator')
        if lhs.is_buffer_access or lhs.is_memview_index:
            if lhs.type.is_pyobject:
                error(self.pos, 'In-place operators not allowed on object buffers in this release.')
            if c_op in ('/', '%') and lhs.type.is_int and (not code.globalstate.directives['cdivision']):
                error(self.pos, 'In-place non-c divide operators not allowed on int buffers.')
            lhs.generate_buffer_setitem_code(rhs, code, c_op)
        elif lhs.is_memview_slice:
            error(self.pos, 'Inplace operators not supported on memoryview slices')
        else:
            code.putln('%s %s= %s;' % (lhs.result(), c_op, rhs.result()))
        lhs.generate_subexpr_disposal_code(code)
        lhs.free_subexpr_temps(code)
        rhs.generate_disposal_code(code)
        rhs.free_temps(code)

    def annotate(self, code):
        self.lhs.annotate(code)
        self.rhs.annotate(code)

    def create_binop_node(self):
        from . import ExprNodes
        return ExprNodes.binop_node(self.pos, self.operator, self.lhs, self.rhs)