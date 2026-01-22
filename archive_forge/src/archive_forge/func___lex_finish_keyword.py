from __future__ import absolute_import
import functools
import json
import re
import sys
def __lex_finish_keyword(self):
    if self.buffer == 'false':
        self.__parser_input(False)
    elif self.buffer == 'true':
        self.__parser_input(True)
    elif self.buffer == 'null':
        self.__parser_input(None)
    else:
        self.__error("invalid keyword '%s'" % self.buffer)