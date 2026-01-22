import bisect
import configparser
import inspect
import io
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from functools import lru_cache
from optparse import OptionParser
def _break_around_binary_operators(tokens):
    """Private function to reduce duplication.

    This factors out the shared details between
    :func:`break_before_binary_operator` and
    :func:`break_after_binary_operator`.
    """
    line_break = False
    unary_context = True
    previous_token_type = None
    previous_text = None
    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.COMMENT:
            continue
        if ('\n' in text or '\r' in text) and token_type != tokenize.STRING:
            line_break = True
        else:
            yield (token_type, text, previous_token_type, previous_text, line_break, unary_context, start)
            unary_context = text in '([{,;'
            line_break = False
            previous_token_type = token_type
            previous_text = text