from __future__ import absolute_import, print_function, division
import locale
from itertools import islice
from collections import defaultdict
from petl.compat import numeric_types, text_type
from petl import config
from petl.util.base import Table
from petl.io.sources import MemorySource
from petl.io.html import tohtml
class See(object):

    def __init__(self, table, limit, vrepr, index_header):
        self.table = table
        self.limit = limit
        self.vrepr = vrepr
        self.index_header = index_header

    def __repr__(self):
        table, overflow = _vis_overflow(self.table, self.limit)
        vrepr = self.vrepr
        index_header = self.index_header
        output = ''
        it = iter(table)
        try:
            flds = next(it)
        except StopIteration:
            return ''
        cols = defaultdict(list)
        for row in it:
            for i, f in enumerate(flds):
                try:
                    cols[str(i)].append(vrepr(row[i]))
                except IndexError:
                    cols[str(f)].append('')
        for i, f in enumerate(flds):
            if index_header:
                f = '%s|%s' % (i, f)
            output += '%s: %s' % (f, ', '.join(cols[str(i)]))
            if overflow:
                output += '...\n'
            else:
                output += '\n'
        return output
    __str__ = __repr__
    __unicode__ = __repr__