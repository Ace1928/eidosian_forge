import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _parse_projection_rhs(self, binding_power):
    if self.BINDING_POWER[self._current_token()] < self._PROJECTION_STOP:
        right = ast.identity()
    elif self._current_token() == 'lbracket':
        right = self._expression(binding_power)
    elif self._current_token() == 'filter':
        right = self._expression(binding_power)
    elif self._current_token() == 'dot':
        self._match('dot')
        right = self._parse_dot_rhs(binding_power)
    else:
        self._raise_parse_error_for_token(self._lookahead_token(0), 'syntax error')
    return right