from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import os
import re
def _Rstrip(line):
    """Strips unescaped trailing spaces."""
    tokens = []
    i = 0
    while i < len(line):
        curr = line[i]
        if curr == '\\':
            if i + 1 >= len(line):
                tokens.append(curr)
                break
            tokens.append(curr + line[i + 1])
            i += 2
        else:
            tokens.append(curr)
            i += 1
    res = []
    only_seen_spaces = True
    for curr in reversed(tokens):
        if only_seen_spaces and curr == ' ':
            continue
        only_seen_spaces = False
        res.append(curr)
    return ''.join(reversed(res))