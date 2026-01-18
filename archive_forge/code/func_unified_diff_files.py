import difflib
import os
import sys
import time
from typing import Type
def unified_diff_files(a, b, sequencematcher=None):
    """Generate the diff for two files."""
    if a == b:
        return []
    if a == '-':
        lines_a = sys.stdin.readlines()
        time_a = time.time()
    else:
        with open(a) as f:
            lines_a = f.readlines()
        time_a = os.stat(a).st_mtime
    if b == '-':
        lines_b = sys.stdin.readlines()
        time_b = time.time()
    else:
        with open(b) as f:
            lines_b = f.readlines()
        time_b = os.stat(b).st_mtime
    return unified_diff(lines_a, lines_b, fromfile=a, tofile=b, sequencematcher=sequencematcher)