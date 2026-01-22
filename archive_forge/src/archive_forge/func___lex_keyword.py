from __future__ import absolute_import
import functools
import json
import re
import sys
def __lex_keyword(self, c):
    if c in Parser.__lex_alpha:
        self.buffer += c
        return True
    else:
        self.__lex_finish_keyword()
        return False