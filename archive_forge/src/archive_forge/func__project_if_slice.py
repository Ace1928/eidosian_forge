import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _project_if_slice(self, left, right):
    index_expr = ast.index_expression([left, right])
    if right['type'] == 'slice':
        return ast.projection(index_expr, self._parse_projection_rhs(self.BINDING_POWER['star']))
    else:
        return index_expr