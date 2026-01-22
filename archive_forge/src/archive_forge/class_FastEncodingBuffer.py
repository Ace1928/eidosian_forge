from ast import parse
import codecs
import collections
import operator
import os
import re
import timeit
from .compat import importlib_metadata_get
class FastEncodingBuffer:
    """a very rudimentary buffer that is faster than StringIO,
    and supports unicode data."""

    def __init__(self, encoding=None, errors='strict'):
        self.data = collections.deque()
        self.encoding = encoding
        self.delim = ''
        self.errors = errors
        self.write = self.data.append

    def truncate(self):
        self.data = collections.deque()
        self.write = self.data.append

    def getvalue(self):
        if self.encoding:
            return self.delim.join(self.data).encode(self.encoding, self.errors)
        else:
            return self.delim.join(self.data)