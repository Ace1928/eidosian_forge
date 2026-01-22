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
class CondExprNode(ExprNode):
    true_val = None
    false_val = None
    is_temp = True
    subexprs = ['test', 'true_val', 'false_val']

    def type_dependencies(self, env):
        return self.true_val.type_dependencies(env) + self.false_val.type_dependencies(env)

    def infer_type(self, env):
        return PyrexTypes.independent_spanning_type(self.true_val.infer_type(env), self.false_val.infer_type(env))

    def calculate_constant_result(self):
        if self.test.constant_result:
            self.constant_result = self.true_val.constant_result
        else:
            self.constant_result = self.false_val.constant_result

    def is_ephemeral(self):
        return self.true_val.is_ephemeral() or self.false_val.is_ephemeral()

    def analyse_types(self, env):
        self.test = self.test.analyse_temp_boolean_expression(env)
        self.true_val = self.true_val.analyse_types(env)
        self.false_val = self.false_val.analyse_types(env)
        return self.analyse_result_type(env)

    def analyse_result_type(self, env):
        true_val_type = self.true_val.type
        false_val_type = self.false_val.type
        self.type = PyrexTypes.independent_spanning_type(true_val_type, false_val_type)
        if self.type.is_reference:
            self.type = PyrexTypes.CFakeReferenceType(self.type.ref_base_type)
        if self.type.is_pyobject:
            self.result_ctype = py_object_type
        elif self.true_val.is_ephemeral() or self.false_val.is_ephemeral():
            error(self.pos, 'Unsafe C derivative of temporary Python reference used in conditional expression')
        if true_val_type.is_pyobject or false_val_type.is_pyobject or self.type.is_pyobject:
            if true_val_type != self.type:
                self.true_val = self.true_val.coerce_to(self.type, env)
            if false_val_type != self.type:
                self.false_val = self.false_val.coerce_to(self.type, env)
        if self.type.is_error:
            self.type_error()
        return self

    def coerce_to_integer(self, env):
        if not self.true_val.type.is_int:
            self.true_val = self.true_val.coerce_to_integer(env)
        if not self.false_val.type.is_int:
            self.false_val = self.false_val.coerce_to_integer(env)
        self.result_ctype = None
        out = self.analyse_result_type(env)
        if not out.type.is_int:
            if out is self:
                out = super(CondExprNode, out).coerce_to_integer(env)
            else:
                out = out.coerce_to_integer(env)
        return out

    def coerce_to(self, dst_type, env):
        if self.true_val.type != dst_type:
            self.true_val = self.true_val.coerce_to(dst_type, env)
        if self.false_val.type != dst_type:
            self.false_val = self.false_val.coerce_to(dst_type, env)
        self.result_ctype = None
        out = self.analyse_result_type(env)
        if out.type != dst_type:
            if out is self:
                out = super(CondExprNode, out).coerce_to(dst_type, env)
            else:
                out = out.coerce_to(dst_type, env)
        return out

    def type_error(self):
        if not (self.true_val.type.is_error or self.false_val.type.is_error):
            error(self.pos, 'Incompatible types in conditional expression (%s; %s)' % (self.true_val.type, self.false_val.type))
        self.type = PyrexTypes.error_type

    def check_const(self):
        return self.test.check_const() and self.true_val.check_const() and self.false_val.check_const()

    def generate_evaluation_code(self, code):
        code.mark_pos(self.pos)
        self.allocate_temp_result(code)
        self.test.generate_evaluation_code(code)
        code.putln('if (%s) {' % self.test.result())
        self.eval_and_get(code, self.true_val)
        code.putln('} else {')
        self.eval_and_get(code, self.false_val)
        code.putln('}')
        self.test.generate_disposal_code(code)
        self.test.free_temps(code)

    def eval_and_get(self, code, expr):
        expr.generate_evaluation_code(code)
        if self.type.is_memoryviewslice:
            expr.make_owned_memoryviewslice(code)
        else:
            expr.make_owned_reference(code)
        code.putln('%s = %s;' % (self.result(), expr.result_as(self.ctype())))
        expr.generate_post_assignment_code(code)
        expr.free_temps(code)

    def generate_subexpr_disposal_code(self, code):
        pass

    def free_subexpr_temps(self, code):
        pass