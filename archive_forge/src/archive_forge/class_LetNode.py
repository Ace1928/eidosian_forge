from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type
class LetNode(Nodes.StatNode, LetNodeMixin):
    child_attrs = ['temp_expression', 'body']

    def __init__(self, lazy_temp, body):
        self.set_temp_expr(lazy_temp)
        self.pos = body.pos
        self.body = body

    def analyse_declarations(self, env):
        self.temp_expression.analyse_declarations(env)
        self.body.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.temp_expression = self.temp_expression.analyse_expressions(env)
        self.body = self.body.analyse_expressions(env)
        return self

    def generate_execution_code(self, code):
        self.setup_temp_expr(code)
        self.body.generate_execution_code(code)
        self.teardown_temp_expr(code)

    def generate_function_definitions(self, env, code):
        self.temp_expression.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)