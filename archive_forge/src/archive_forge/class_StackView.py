from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
class StackView(Table):

    def __init__(self, sources, missing=None, trim=True, pad=True):
        self.sources = sources
        self.missing = missing
        self.trim = trim
        self.pad = pad

    def __iter__(self):
        return iterstack(self.sources, self.missing, self.trim, self.pad)