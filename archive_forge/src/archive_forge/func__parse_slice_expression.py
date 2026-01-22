import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _parse_slice_expression(self):
    parts = [None, None, None]
    index = 0
    current_token = self._current_token()
    while not current_token == 'rbracket' and index < 3:
        if current_token == 'colon':
            index += 1
            if index == 3:
                self._raise_parse_error_for_token(self._lookahead_token(0), 'syntax error')
            self._advance()
        elif current_token == 'number':
            parts[index] = self._lookahead_token(0)['value']
            self._advance()
        else:
            self._raise_parse_error_for_token(self._lookahead_token(0), 'syntax error')
        current_token = self._current_token()
    self._match('rbracket')
    return ast.slice(*parts)