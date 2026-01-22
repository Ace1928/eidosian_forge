from __future__ import absolute_import, print_function, division
import locale
from itertools import islice
from collections import defaultdict
from petl.compat import numeric_types, text_type
from petl import config
from petl.util.base import Table
from petl.io.sources import MemorySource
from petl.io.html import tohtml
class Look(object):

    def __init__(self, table, limit, vrepr, index_header, style, truncate, width):
        self.table = table
        self.limit = limit
        self.vrepr = vrepr
        self.index_header = index_header
        self.style = style
        self.truncate = truncate
        self.width = width

    def __repr__(self):
        table, overflow = _vis_overflow(self.table, self.limit)
        style = self.style
        vrepr = self.vrepr
        index_header = self.index_header
        truncate = self.truncate
        width = self.width
        if style == 'simple':
            output = _look_simple(table, vrepr=vrepr, index_header=index_header, truncate=truncate, width=width)
        elif style == 'minimal':
            output = _look_minimal(table, vrepr=vrepr, index_header=index_header, truncate=truncate, width=width)
        else:
            output = _look_grid(table, vrepr=vrepr, index_header=index_header, truncate=truncate, width=width)
        if overflow:
            output += '...\n'
        return output
    __str__ = __repr__
    __unicode__ = __repr__