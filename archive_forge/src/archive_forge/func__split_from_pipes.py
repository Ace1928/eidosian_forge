import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _split_from_pipes(self, row):
    _, separator, rest = self._pipe_splitter.split(row, 1)
    yield separator
    while self._pipe_splitter.search(rest):
        cell, separator, rest = self._pipe_splitter.split(rest, 1)
        yield cell
        yield separator
    yield rest