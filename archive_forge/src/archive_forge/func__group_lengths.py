import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _group_lengths(grouping):
    """Convert a localeconv-style grouping into a (possibly infinite)
    iterable of integers representing group lengths.

    """
    from itertools import chain, repeat
    if not grouping:
        return []
    elif grouping[-1] == 0 and len(grouping) >= 2:
        return chain(grouping[:-1], repeat(grouping[-2]))
    elif grouping[-1] == _locale.CHAR_MAX:
        return grouping[:-1]
    else:
        raise ValueError('unrecognised format for grouping')