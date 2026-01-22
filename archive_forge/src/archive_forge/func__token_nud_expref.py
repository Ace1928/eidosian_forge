import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _token_nud_expref(self, token):
    expression = self._expression(self.BINDING_POWER['expref'])
    return ast.expref(expression)