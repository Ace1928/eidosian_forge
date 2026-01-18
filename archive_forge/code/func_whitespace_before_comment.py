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
def whitespace_before_comment(logical_line, tokens):
    """Separate inline comments by at least two spaces.

    An inline comment is a comment on the same line as a statement.  Inline
    comments should be separated by at least two spaces from the statement.
    They should start with a # and a single space.

    Each line of a block comment starts with a # and a single space
    (unless it is indented text inside the comment).

    Okay: x = x + 1  # Increment x
    Okay: x = x + 1    # Increment x
    Okay: # Block comment
    E261: x = x + 1 # Increment x
    E262: x = x + 1  #Increment x
    E262: x = x + 1  #  Increment x
    E265: #Block comment
    E266: ### Block comment
    """
    prev_end = (0, 0)
    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.COMMENT:
            inline_comment = line[:start[1]].strip()
            if inline_comment:
                if prev_end[0] == start[0] and start[1] < prev_end[1] + 2:
                    yield (prev_end, 'E261 at least two spaces before inline comment')
            symbol, sp, comment = text.partition(' ')
            bad_prefix = symbol not in '#:' and (symbol.lstrip('#')[:1] or '#')
            if inline_comment:
                if bad_prefix or comment[:1] in WHITESPACE:
                    yield (start, "E262 inline comment should start with '# '")
            elif bad_prefix and (bad_prefix != '!' or start[0] > 1):
                if bad_prefix != '#':
                    yield (start, "E265 block comment should start with '# '")
                elif comment:
                    yield (start, "E266 too many leading '#' for block comment")
        elif token_type != tokenize.NL:
            prev_end = end