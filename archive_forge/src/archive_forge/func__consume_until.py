import string
import warnings
from json import loads
from jmespath.exceptions import LexerError, EmptyExpressionError
def _consume_until(self, delimiter):
    start = self._position
    buff = ''
    self._next()
    while self._current != delimiter:
        if self._current == '\\':
            buff += '\\'
            self._next()
        if self._current is None:
            raise LexerError(lexer_position=start, lexer_value=self._expression[start:], message='Unclosed %s delimiter' % delimiter)
        buff += self._current
        self._next()
    self._next()
    return buff