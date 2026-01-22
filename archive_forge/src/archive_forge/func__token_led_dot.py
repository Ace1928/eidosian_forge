import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _token_led_dot(self, left):
    if not self._current_token() == 'star':
        right = self._parse_dot_rhs(self.BINDING_POWER['dot'])
        if left['type'] == 'subexpression':
            left['children'].append(right)
            return left
        else:
            return ast.subexpression([left, right])
    else:
        self._advance()
        right = self._parse_projection_rhs(self.BINDING_POWER['dot'])
        return ast.value_projection(left, right)