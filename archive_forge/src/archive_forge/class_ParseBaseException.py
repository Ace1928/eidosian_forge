import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class ParseBaseException(Exception):
    """base exception class for all parsing runtime exceptions"""

    def __init__(self, pstr, loc=0, msg=None, elem=None):
        self.loc = loc
        if msg is None:
            self.msg = pstr
            self.pstr = ''
        else:
            self.msg = msg
            self.pstr = pstr
        self.parserElement = elem

    def __getattr__(self, aname):
        """supported attributes by name are:
            - lineno - returns the line number of the exception text
            - col - returns the column number of the exception text
            - line - returns the line containing the exception text
        """
        if aname == 'lineno':
            return lineno(self.loc, self.pstr)
        elif aname in ('col', 'column'):
            return col(self.loc, self.pstr)
        elif aname == 'line':
            return line(self.loc, self.pstr)
        else:
            raise AttributeError(aname)

    def __str__(self):
        return '%s (at char %d), (line:%d, col:%d)' % (self.msg, self.loc, self.lineno, self.column)

    def __repr__(self):
        return _ustr(self)

    def markInputline(self, markerString='>!<'):
        """Extracts the exception line from the input string, and marks
           the location of the exception with a special symbol.
        """
        line_str = self.line
        line_column = self.column - 1
        if markerString:
            line_str = ''.join(line_str[:line_column], markerString, line_str[line_column:])
        return line_str.strip()

    def __dir__(self):
        return 'loc msg pstr parserElement lineno col line markInputline __str__ __repr__'.split()