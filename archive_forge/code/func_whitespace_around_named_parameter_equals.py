from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def whitespace_around_named_parameter_equals(logical_line, tokens):
    """Don't use spaces around the '=' sign in function arguments.

    Don't use spaces around the '=' sign when used to indicate a
    keyword argument or a default parameter value.

    Okay: def complex(real, imag=0.0):
    Okay: return magic(r=real, i=imag)
    Okay: boolean(a == b)
    Okay: boolean(a != b)
    Okay: boolean(a <= b)
    Okay: boolean(a >= b)
    Okay: def foo(arg: int = 42):
    Okay: async def foo(arg: int = 42):

    E251: def complex(real, imag = 0.0):
    E251: return magic(r = real, i = imag)
    """
    parens = 0
    no_space = False
    prev_end = None
    annotated_func_arg = False
    in_def = logical_line.startswith(('def', 'async def'))
    message = 'E251 unexpected spaces around keyword / parameter equals'
    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.NL:
            continue
        if no_space:
            no_space = False
            if start != prev_end:
                yield (prev_end, message)
        if token_type == tokenize.OP:
            if text in '([':
                parens += 1
            elif text in ')]':
                parens -= 1
            elif in_def and text == ':' and (parens == 1):
                annotated_func_arg = True
            elif parens and text == ',' and (parens == 1):
                annotated_func_arg = False
            elif parens and text == '=' and (not annotated_func_arg):
                no_space = True
                if start != prev_end:
                    yield (prev_end, message)
            if not parens:
                annotated_func_arg = False
        prev_end = end