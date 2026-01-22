from __future__ import absolute_import
import functools
import json
import re
import sys
def __lex_start_number(self, c):
    self.buffer = c
    self.lex_state = Parser.__lex_number