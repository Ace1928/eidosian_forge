import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _match_multiple_tokens(self, token_types):
    if self._current_token() not in token_types:
        self._raise_parse_error_maybe_eof(token_types, self._lookahead_token(0))
    self._advance()