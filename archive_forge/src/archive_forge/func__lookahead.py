import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _lookahead(self, number):
    return self._tokens[self._index + number]['type']