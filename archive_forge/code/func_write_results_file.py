import io
import linecache
import os
import sys
import sysconfig
import token
import tokenize
import inspect
import gc
import dis
import pickle
from time import monotonic as _time
import threading
def write_results_file(self, path, lines, lnotab, lines_hit, encoding=None):
    """Return a coverage results file in path."""
    try:
        outfile = open(path, 'w', encoding=encoding)
    except OSError as err:
        print('trace: Could not open %r for writing: %s - skipping' % (path, err), file=sys.stderr)
        return (0, 0)
    n_lines = 0
    n_hits = 0
    with outfile:
        for lineno, line in enumerate(lines, 1):
            if lineno in lines_hit:
                outfile.write('%5d: ' % lines_hit[lineno])
                n_hits += 1
                n_lines += 1
            elif lineno in lnotab and (not PRAGMA_NOCOVER in line):
                outfile.write('>>>>>> ')
                n_lines += 1
            else:
                outfile.write('       ')
            outfile.write(line.expandtabs(8))
    return (n_hits, n_lines)