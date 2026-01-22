import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def _grouping_intervals(grouping):
    last_interval = None
    for interval in grouping:
        if interval == CHAR_MAX:
            return
        if interval == 0:
            if last_interval is None:
                raise ValueError('invalid grouping')
            while True:
                yield last_interval
        yield interval
        last_interval = interval