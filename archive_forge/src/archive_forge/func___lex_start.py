from __future__ import absolute_import
import functools
import json
import re
import sys
def __lex_start(self, c):
    Parser.__lex_start_actions.get(c, Parser.__lex_start_error)(self, c)
    return True