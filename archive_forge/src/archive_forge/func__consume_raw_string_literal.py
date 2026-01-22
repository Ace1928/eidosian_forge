import string
import warnings
from json import loads
from jmespath.exceptions import LexerError, EmptyExpressionError
def _consume_raw_string_literal(self):
    start = self._position
    lexeme = self._consume_until("'").replace("\\'", "'")
    token_len = self._position - start
    return {'type': 'literal', 'value': lexeme, 'start': start, 'end': token_len}