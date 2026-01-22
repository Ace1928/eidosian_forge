from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
class AddFieldUsingContextView(Table):

    def __init__(self, table, field, query):
        self.table = table
        self.field = field
        self.query = query

    def __iter__(self):
        return iteraddfieldusingcontext(self.table, self.field, self.query)