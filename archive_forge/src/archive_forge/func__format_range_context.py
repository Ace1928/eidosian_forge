from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _format_range_context(start, stop):
    """Convert range to the "ed" format"""
    beginning = start + 1
    length = stop - start
    if not length:
        beginning -= 1
    if length <= 1:
        return '{}'.format(beginning)
    return '{},{}'.format(beginning, beginning + length - 1)