from __future__ import absolute_import
import functools
import json
import re
import sys
def __lex_number(self, c):
    if c in '.0123456789eE-+':
        self.buffer += c
        return True
    else:
        self.__lex_finish_number()
        return False