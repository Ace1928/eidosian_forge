import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _token_nud_lbrace(self, token):
    return self._parse_multi_select_hash()