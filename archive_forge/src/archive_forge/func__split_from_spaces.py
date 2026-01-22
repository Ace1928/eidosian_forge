import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _split_from_spaces(self, row):
    yield ''
    for value in self._space_splitter.split(row):
        yield value