from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type
class ResultRefNode(AtomicExprNode):
    subexprs = []
    lhs_of_first_assignment = False

    def __init__(self, expression=None, pos=None, type=None, may_hold_none=True, is_temp=False):
        self.expression = expression
        self.pos = None
        self.may_hold_none = may_hold_none
        if expression is not None:
            self.pos = expression.pos
            self.type = getattr(expression, 'type', None)
        if pos is not None:
            self.pos = pos
        if type is not None:
            self.type = type
        if is_temp:
            self.is_temp = True
        assert self.pos is not None

    def clone_node(self):
        return self

    def type_dependencies(self, env):
        if self.expression:
            return self.expression.type_dependencies(env)
        else:
            return ()

    def update_expression(self, expression):
        self.expression = expression
        type = getattr(expression, 'type', None)
        if type:
            self.type = type

    def analyse_target_declaration(self, env):
        pass

    def analyse_types(self, env):
        if self.expression is not None:
            if not self.expression.type:
                self.expression = self.expression.analyse_types(env)
            self.type = self.expression.type
        return self

    def infer_type(self, env):
        if self.type is not None:
            return self.type
        if self.expression is not None:
            if self.expression.type is not None:
                return self.expression.type
            return self.expression.infer_type(env)
        assert False, 'cannot infer type of ResultRefNode'

    def may_be_none(self):
        if not self.type.is_pyobject:
            return False
        return self.may_hold_none

    def _DISABLED_may_be_none(self):
        if self.expression is not None:
            return self.expression.may_be_none()
        if self.type is not None:
            return self.type.is_pyobject
        return True

    def is_simple(self):
        return True

    def result(self):
        try:
            return self.result_code
        except AttributeError:
            if self.expression is not None:
                self.result_code = self.expression.result()
        return self.result_code

    def generate_evaluation_code(self, code):
        pass

    def generate_result_code(self, code):
        pass

    def generate_disposal_code(self, code):
        pass

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False):
        if self.type.is_pyobject:
            rhs.make_owned_reference(code)
            if not self.lhs_of_first_assignment:
                code.put_decref(self.result(), self.ctype())
        code.putln('%s = %s;' % (self.result(), rhs.result() if overloaded_assignment else rhs.result_as(self.ctype())))
        rhs.generate_post_assignment_code(code)
        rhs.free_temps(code)

    def allocate_temps(self, env):
        pass

    def release_temp(self, env):
        pass

    def free_temps(self, code):
        pass