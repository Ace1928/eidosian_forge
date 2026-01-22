from __future__ import absolute_import
import functools
import json
import re
import sys
def __lex_4hex(self, s):
    if len(s) < 4:
        self.__error('quoted string ends within \\u escape')
    elif not Parser.__4hex_re.match(s):
        self.__error('malformed \\u escape')
    elif s == '0000':
        self.__error('null bytes not supported in quoted strings')
    else:
        return int(s, 16)