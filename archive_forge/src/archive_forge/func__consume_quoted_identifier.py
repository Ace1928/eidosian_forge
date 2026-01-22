import string
import warnings
from json import loads
from jmespath.exceptions import LexerError, EmptyExpressionError
def _consume_quoted_identifier(self):
    start = self._position
    lexeme = '"' + self._consume_until('"') + '"'
    try:
        token_len = self._position - start
        return {'type': 'quoted_identifier', 'value': loads(lexeme), 'start': start, 'end': token_len}
    except ValueError as e:
        error_message = str(e).split(':')[0]
        raise LexerError(lexer_position=start, lexer_value=lexeme, message=error_message)