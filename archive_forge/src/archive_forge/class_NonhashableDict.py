from fontTools.misc.timeTools import timestampNow
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from functools import reduce
import operator
import logging
class NonhashableDict(object):
    """A dictionary-like object mapping objects to values."""

    def __init__(self, keys, values=None):
        if values is None:
            self.d = {id(v): i for i, v in enumerate(keys)}
        else:
            self.d = {id(k): v for k, v in zip(keys, values)}

    def __getitem__(self, k):
        return self.d[id(k)]

    def __setitem__(self, k, v):
        self.d[id(k)] = v

    def __delitem__(self, k):
        del self.d[id(k)]