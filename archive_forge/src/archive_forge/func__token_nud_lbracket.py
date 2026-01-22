import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _token_nud_lbracket(self, token):
    if self._current_token() in ['number', 'colon']:
        right = self._parse_index_expression()
        return self._project_if_slice(ast.identity(), right)
    elif self._current_token() == 'star' and self._lookahead(1) == 'rbracket':
        self._advance()
        self._advance()
        right = self._parse_projection_rhs(self.BINDING_POWER['star'])
        return ast.projection(ast.identity(), right)
    else:
        return self._parse_multi_select_list()