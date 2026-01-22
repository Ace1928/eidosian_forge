from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class SourceFileDescriptor(object):

    def __init__(self, filename, lexer, formatter=None):
        self.filename = filename
        self.lexer = lexer
        self.formatter = formatter

    def valid(self):
        return self.filename is not None

    def lex(self, code):
        if pygments and self.lexer and parameters.colorize_code:
            bg = parameters.terminal_background.value
            if self.formatter is None:
                formatter = pygments.formatters.TerminalFormatter(bg=bg)
            else:
                formatter = self.formatter
            return pygments.highlight(code, self.lexer, formatter)
        return code

    def _get_source(self, start, stop, lex_source, mark_line, lex_entire):
        with open(self.filename) as f:
            if lex_source and lex_entire:
                f = self.lex(f.read()).splitlines()
            slice = itertools.islice(f, start - 1, stop - 1)
            for idx, line in enumerate(slice):
                if start + idx == mark_line:
                    prefix = '>'
                else:
                    prefix = ' '
                if lex_source and (not lex_entire):
                    line = self.lex(line)
                yield ('%s %4d    %s' % (prefix, start + idx, line.rstrip()))

    def get_source(self, start, stop=None, lex_source=True, mark_line=0, lex_entire=False):
        exc = gdb.GdbError('Unable to retrieve source code')
        if not self.filename:
            raise exc
        start = max(start, 1)
        if stop is None:
            stop = start + 1
        try:
            return '\n'.join(self._get_source(start, stop, lex_source, mark_line, lex_entire))
        except IOError:
            raise exc