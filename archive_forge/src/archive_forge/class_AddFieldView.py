from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
class AddFieldView(Table):

    def __init__(self, source, field, value=None, index=None, missing=None):
        self.source = stack(source, missing=missing)
        self.field = field
        self.value = value
        self.index = index

    def __iter__(self):
        return iteraddfield(self.source, self.field, self.value, self.index)