import string
import warnings
from json import loads
from jmespath.exceptions import LexerError, EmptyExpressionError
def _consume_number(self):
    start = self._position
    buff = self._current
    while self._next() in self.VALID_NUMBER:
        buff += self._current
    return buff