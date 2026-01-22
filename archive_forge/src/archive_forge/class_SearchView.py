from __future__ import absolute_import, print_function, division
import re
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.util.base import Table, asindices
from petl.transform.basics import TransformError
from petl.transform.conversions import convert
class SearchView(Table):

    def __init__(self, table, pattern, field=None, flags=0, complement=False):
        self.table = table
        self.pattern = pattern
        self.field = field
        self.flags = flags
        self.complement = complement

    def __iter__(self):
        return itersearch(self.table, self.pattern, self.field, self.flags, self.complement)