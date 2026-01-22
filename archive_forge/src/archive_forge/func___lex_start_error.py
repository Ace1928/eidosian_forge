from __future__ import absolute_import
import functools
import json
import re
import sys
def __lex_start_error(self, c):
    if ord(c) >= 32 and ord(c) < 128:
        self.__error("invalid character '%s'" % c)
    else:
        self.__error('invalid character U+%04x' % ord(c))