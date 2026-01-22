import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _token_nud_flatten(self, token):
    left = ast.flatten(ast.identity())
    right = self._parse_projection_rhs(self.BINDING_POWER['flatten'])
    return ast.projection(left, right)