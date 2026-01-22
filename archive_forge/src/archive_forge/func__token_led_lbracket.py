import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _token_led_lbracket(self, left):
    token = self._lookahead_token(0)
    if token['type'] in ['number', 'colon']:
        right = self._parse_index_expression()
        if left['type'] == 'index_expression':
            left['children'].append(right)
            return left
        else:
            return self._project_if_slice(left, right)
    else:
        self._match('star')
        self._match('rbracket')
        right = self._parse_projection_rhs(self.BINDING_POWER['star'])
        return ast.projection(left, right)