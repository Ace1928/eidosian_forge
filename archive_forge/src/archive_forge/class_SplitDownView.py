from __future__ import absolute_import, print_function, division
import re
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.util.base import Table, asindices
from petl.transform.basics import TransformError
from petl.transform.conversions import convert
class SplitDownView(Table):

    def __init__(self, table, field, pattern, maxsplit=0, flags=0):
        self.table = table
        self.field = field
        self.pattern = pattern
        self.maxsplit = maxsplit
        self.flags = flags

    def __iter__(self):
        return itersplitdown(self.table, self.field, self.pattern, self.maxsplit, self.flags)