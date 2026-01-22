from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _line_wrapper(self, diffs):
    """Returns iterator that splits (wraps) mdiff text lines"""
    for fromdata, todata, flag in diffs:
        if flag is None:
            yield (fromdata, todata, flag)
            continue
        (fromline, fromtext), (toline, totext) = (fromdata, todata)
        fromlist, tolist = ([], [])
        self._split_line(fromlist, fromline, fromtext)
        self._split_line(tolist, toline, totext)
        while fromlist or tolist:
            if fromlist:
                fromdata = fromlist.pop(0)
            else:
                fromdata = ('', ' ')
            if tolist:
                todata = tolist.pop(0)
            else:
                todata = ('', ' ')
            yield (fromdata, todata, flag)