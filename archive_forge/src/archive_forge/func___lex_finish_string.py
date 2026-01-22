from __future__ import absolute_import
import functools
import json
import re
import sys
def __lex_finish_string(self):
    inp = self.buffer
    out = u''
    while len(inp):
        backslash = inp.find('\\')
        if backslash == -1:
            out += inp
            break
        out += inp[:backslash]
        inp = inp[backslash + 1:]
        if inp == '':
            self.__error('quoted string may not end with backslash')
            return
        replacement = Parser.__unescape.get(inp[0])
        if replacement is not None:
            out += replacement
            inp = inp[1:]
            continue
        elif inp[0] != u'u':
            self.__error('bad escape \\%s' % inp[0])
            return
        c0 = self.__lex_4hex(inp[1:5])
        if c0 is None:
            return
        inp = inp[5:]
        if Parser.__is_leading_surrogate(c0):
            if inp[:2] != u'\\u':
                self.__error('malformed escaped surrogate pair')
                return
            c1 = self.__lex_4hex(inp[2:6])
            if c1 is None:
                return
            if not Parser.__is_trailing_surrogate(c1):
                self.__error('second half of escaped surrogate pair is not trailing surrogate')
                return
            code_point = Parser.__utf16_decode_surrogate_pair(c0, c1)
            inp = inp[6:]
        else:
            code_point = c0
        out += chr(code_point)
    self.__parser_input('string', out)