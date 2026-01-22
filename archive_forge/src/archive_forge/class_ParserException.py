import os
import re
import sys
import traceback
import ast
import importlib
from re import sub, findall
from types import CodeType
from functools import partial
from collections import OrderedDict, defaultdict
import kivy.lang.builder  # imported as absolute to avoid circular import
from kivy.logger import Logger
from kivy.cache import Cache
from kivy import require
from kivy.resources import resource_find
from kivy.utils import rgba
import kivy.metrics as Metrics
class ParserException(Exception):
    """Exception raised when something wrong happened in a kv file.
    """

    def __init__(self, context, line, message, cause=None):
        self.filename = context.filename or '<inline>'
        self.line = line
        sourcecode = context.sourcecode
        sc_start = max(0, line - 2)
        sc_stop = min(len(sourcecode), line + 3)
        sc = ['...']
        for x in range(sc_start, sc_stop):
            if x == line:
                sc += ['>> %4d:%s' % (line + 1, sourcecode[line][1])]
            else:
                sc += ['   %4d:%s' % (x + 1, sourcecode[x][1])]
        sc += ['...']
        sc = '\n'.join(sc)
        message = 'Parser: File "%s", line %d:\n%s\n%s' % (self.filename, self.line + 1, sc, message)
        if cause:
            message += '\n' + ''.join(traceback.format_tb(cause))
        super(ParserException, self).__init__(message)