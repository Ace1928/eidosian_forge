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
def trailing_whitespace(physical_line):
    """Trailing whitespace is superfluous.

    The warning returned varies on whether the line itself is blank, for easier
    filtering for those who want to indent their blank lines.

    Okay: spam(1)\\n#
    W291: spam(1) \\n#
    W293: class Foo(object):\\n    \\n    bang = 12
    """
    physical_line = physical_line.rstrip('\n')
    physical_line = physical_line.rstrip('\r')
    physical_line = physical_line.rstrip('\x0c')
    stripped = physical_line.rstrip(' \t\x0b')
    if physical_line != stripped:
        if stripped:
            return (len(stripped), 'W291 trailing whitespace')
        else:
            return (0, 'W293 blank line contains whitespace')