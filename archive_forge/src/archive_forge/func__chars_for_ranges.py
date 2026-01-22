import sys
from itertools import filterfalse
from typing import List, Tuple, Union
@_lazyclassproperty
def _chars_for_ranges(cls):
    ret = []
    for cc in cls.__mro__:
        if cc is unicode_set:
            break
        for rr in getattr(cc, '_ranges', ()):
            ret.extend(range(rr[0], rr[-1] + 1))
    return [chr(c) for c in sorted(set(ret))]