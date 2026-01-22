import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class RowSplitter(object):
    _space_splitter = re.compile('( {2,})')
    _pipe_splitter = re.compile('((?:^| +)\\|(?: +|$))')

    def split(self, row):
        splitter = row.startswith('| ') and self._split_from_pipes or self._split_from_spaces
        for value in splitter(row):
            yield value
        yield '\n'

    def _split_from_spaces(self, row):
        yield ''
        for value in self._space_splitter.split(row):
            yield value

    def _split_from_pipes(self, row):
        _, separator, rest = self._pipe_splitter.split(row, 1)
        yield separator
        while self._pipe_splitter.search(rest):
            cell, separator, rest = self._pipe_splitter.split(rest, 1)
            yield cell
            yield separator
        yield rest