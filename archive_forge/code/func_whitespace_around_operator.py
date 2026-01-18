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
def whitespace_around_operator(logical_line):
    """Avoid extraneous whitespace around an operator.

    Okay: a = 12 + 3
    E221: a = 4  + 5
    E222: a = 4 +  5
    E223: a = 4\\t+ 5
    E224: a = 4 +\\t5
    """
    for match in OPERATOR_REGEX.finditer(logical_line):
        before, after = match.groups()
        if '\t' in before:
            yield (match.start(1), 'E223 tab before operator')
        elif len(before) > 1:
            yield (match.start(1), 'E221 multiple spaces before operator')
        if '\t' in after:
            yield (match.start(2), 'E224 tab after operator')
        elif len(after) > 1:
            yield (match.start(2), 'E222 multiple spaces after operator')