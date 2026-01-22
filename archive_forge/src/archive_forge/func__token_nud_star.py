import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _token_nud_star(self, token):
    left = ast.identity()
    if self._current_token() == 'rbracket':
        right = ast.identity()
    else:
        right = self._parse_projection_rhs(self.BINDING_POWER['star'])
    return ast.value_projection(left, right)