from collections import defaultdict
import copy
import itertools
import os
import linecache
import pprint
import re
import sys
import operator
from types import FunctionType, BuiltinFunctionType
from functools import total_ordering
from io import StringIO
from numba.core import errors, config
from numba.core.utils import (BINOPS_TO_OPERATORS, INPLACE_BINOPS_TO_OPERATORS,
from numba.core.errors import (NotDefinedError, RedefinedError,
from numba.core import consts
class Loc(object):
    """Source location

    """
    _defmatcher = re.compile('def\\s+(\\w+)\\(.*')

    def __init__(self, filename, line, col=None, maybe_decorator=False):
        """ Arguments:
        filename - name of the file
        line - line in file
        col - column
        maybe_decorator - Set to True if location is likely a jit decorator
        """
        self.filename = filename
        self.line = line
        self.col = col
        self.lines = None
        self.maybe_decorator = maybe_decorator

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        if self.filename != other.filename:
            return False
        if self.line != other.line:
            return False
        if self.col != other.col:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def from_function_id(cls, func_id):
        return cls(func_id.filename, func_id.firstlineno, maybe_decorator=True)

    def __repr__(self):
        return 'Loc(filename=%s, line=%s, col=%s)' % (self.filename, self.line, self.col)

    def __str__(self):
        if self.col is not None:
            return '%s (%s:%s)' % (self.filename, self.line, self.col)
        else:
            return '%s (%s)' % (self.filename, self.line)

    def _find_definition(self):
        fn_name = None
        lines = self.get_lines()
        for x in reversed(lines[:self.line - 1]):
            if x.strip().startswith('def '):
                fn_name = x
                break
        return fn_name

    def _raw_function_name(self):
        defn = self._find_definition()
        if defn:
            return self._defmatcher.match(defn.strip()).groups()[0]
        else:
            return None

    def get_lines(self):
        if self.lines is None:
            self.lines = linecache.getlines(self._get_path())
        return self.lines

    def _get_path(self):
        path = None
        try:
            path = os.path.relpath(self.filename)
        except ValueError:
            path = os.path.abspath(self.filename)
        return path

    def strformat(self, nlines_up=2):
        lines = self.get_lines()
        use_line = self.line
        if self.maybe_decorator:
            tmplines = [''] + lines
            if lines and use_line and ('def ' not in tmplines[use_line]):
                min_line = max(0, use_line)
                max_line = use_line + 10
                selected = tmplines[min_line:max_line]
                index = 0
                for idx, x in enumerate(selected):
                    if 'def ' in x:
                        index = idx
                        break
                use_line = use_line + index
        ret = []
        if lines and use_line > 0:

            def count_spaces(string):
                spaces = 0
                for x in itertools.takewhile(str.isspace, str(string)):
                    spaces += 1
                return spaces
            selected = lines[max(0, use_line - nlines_up):use_line]
            def_found = False
            for x in selected:
                if 'def ' in x:
                    def_found = True
            if not def_found:
                fn_name = None
                for x in reversed(lines[:use_line - 1]):
                    if 'def ' in x:
                        fn_name = x
                        break
                if fn_name:
                    ret.append(fn_name)
                    spaces = count_spaces(x)
                    ret.append(' ' * (4 + spaces) + '<source elided>\n')
            if selected:
                ret.extend(selected[:-1])
                ret.append(_termcolor.highlight(selected[-1]))
                spaces = count_spaces(selected[-1])
                ret.append(' ' * spaces + _termcolor.indicate('^'))
        if not ret:
            if not lines:
                ret = '<source missing, REPL/exec in use?>'
            elif use_line <= 0:
                ret = '<source line number missing>'
        err = _termcolor.filename('\nFile "%s", line %d:') + '\n%s'
        tmp = err % (self._get_path(), use_line, _termcolor.code(''.join(ret)))
        return tmp

    def with_lineno(self, line, col=None):
        """
        Return a new Loc with this line number.
        """
        return type(self)(self.filename, line, col)

    def short(self):
        """
        Returns a short string
        """
        shortfilename = os.path.basename(self.filename)
        return '%s:%s' % (shortfilename, self.line)