from builtins import open as _builtin_open
from codecs import lookup, BOM_UTF8
import collections
import functools
from io import TextIOWrapper
import itertools as _itertools
import re
import sys
from token import *
from token import EXACT_TOKEN_TYPES
import token
def untokenize(self, iterable):
    it = iter(iterable)
    indents = []
    startline = False
    for t in it:
        if len(t) == 2:
            self.compat(t, it)
            break
        tok_type, token, start, end, line = t
        if tok_type == ENCODING:
            self.encoding = token
            continue
        if tok_type == ENDMARKER:
            break
        if tok_type == INDENT:
            indents.append(token)
            continue
        elif tok_type == DEDENT:
            indents.pop()
            self.prev_row, self.prev_col = end
            continue
        elif tok_type in (NEWLINE, NL):
            startline = True
        elif startline and indents:
            indent = indents[-1]
            if start[1] >= len(indent):
                self.tokens.append(indent)
                self.prev_col = len(indent)
            startline = False
        self.add_whitespace(start)
        self.tokens.append(token)
        self.prev_row, self.prev_col = end
        if tok_type in (NEWLINE, NL):
            self.prev_row += 1
            self.prev_col = 0
    return ''.join(self.tokens)