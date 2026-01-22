import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def _format_final_exc_line(etype, value):
    valuestr = _safe_string(value, 'exception')
    if value is None or not valuestr:
        line = '%s\n' % etype
    else:
        line = '%s: %s\n' % (etype, valuestr)
    return line