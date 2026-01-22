import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _error_nud_token(self, token):
    if token['type'] == 'eof':
        raise exceptions.IncompleteExpressionError(token['start'], token['value'], token['type'])
    self._raise_parse_error_for_token(token, 'invalid token')