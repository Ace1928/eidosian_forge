from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type
def analyse_expressions(self, env):
    self.temp_expression = self.temp_expression.analyse_expressions(env)
    self.body = self.body.analyse_expressions(env)
    return self