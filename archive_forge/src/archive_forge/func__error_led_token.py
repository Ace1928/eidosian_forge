import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _error_led_token(self, token):
    self._raise_parse_error_for_token(token, 'invalid token')