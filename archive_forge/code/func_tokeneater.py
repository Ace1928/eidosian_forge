import abc
import ast
import dis
import collections.abc
import enum
import importlib.machinery
import itertools
import linecache
import os
import re
import sys
import tokenize
import token
import types
import functools
import builtins
from keyword import iskeyword
from operator import attrgetter
from collections import namedtuple, OrderedDict
def tokeneater(self, type, token, srowcol, erowcol, line):
    if not self.started and (not self.indecorator):
        if token == '@':
            self.indecorator = True
        elif token in ('def', 'class', 'lambda'):
            if token == 'lambda':
                self.islambda = True
            self.started = True
        self.passline = True
    elif type == tokenize.NEWLINE:
        self.passline = False
        self.last = srowcol[0]
        if self.islambda:
            raise EndOfBlock
        if self.indecorator:
            self.indecorator = False
    elif self.passline:
        pass
    elif type == tokenize.INDENT:
        if self.body_col0 is None and self.started:
            self.body_col0 = erowcol[1]
        self.indent = self.indent + 1
        self.passline = True
    elif type == tokenize.DEDENT:
        self.indent = self.indent - 1
        if self.indent <= 0:
            raise EndOfBlock
    elif type == tokenize.COMMENT:
        if self.body_col0 is not None and srowcol[1] >= self.body_col0:
            self.last = srowcol[0]
    elif self.indent == 0 and type not in (tokenize.COMMENT, tokenize.NL):
        raise EndOfBlock