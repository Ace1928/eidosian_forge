from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import INVALID_TOKEN_TYPE
from antlr3.tokens import CommonToken
from antlr3.tree import CommonTree, CommonTreeAdaptor
import six
from six.moves import range
class TreePatternLexer(object):

    def __init__(self, pattern):
        self.pattern = pattern
        self.p = -1
        self.c = None
        self.n = len(pattern)
        self.sval = None
        self.error = False
        self.consume()
    __idStartChar = frozenset('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_')
    __idChar = __idStartChar | frozenset('0123456789')

    def nextToken(self):
        self.sval = ''
        while self.c != EOF:
            if self.c in (' ', '\n', '\r', '\t'):
                self.consume()
                continue
            if self.c in self.__idStartChar:
                self.sval += self.c
                self.consume()
                while self.c in self.__idChar:
                    self.sval += self.c
                    self.consume()
                return ID
            if self.c == '(':
                self.consume()
                return BEGIN
            if self.c == ')':
                self.consume()
                return END
            if self.c == '%':
                self.consume()
                return PERCENT
            if self.c == ':':
                self.consume()
                return COLON
            if self.c == '.':
                self.consume()
                return DOT
            if self.c == '[':
                self.consume()
                while self.c != ']':
                    if self.c == '\\':
                        self.consume()
                        if self.c != ']':
                            self.sval += '\\'
                        self.sval += self.c
                    else:
                        self.sval += self.c
                    self.consume()
                self.consume()
                return ARG
            self.consume()
            self.error = True
            return EOF
        return EOF

    def consume(self):
        self.p += 1
        if self.p >= self.n:
            self.c = EOF
        else:
            self.c = self.pattern[self.p]