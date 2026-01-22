from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
class MoveFieldView(Table):

    def __init__(self, table, field, index, missing=None):
        self.table = table
        self.field = field
        self.index = index
        self.missing = missing

    def __iter__(self):
        it = iter(self.table)
        try:
            hdr = next(it)
        except StopIteration:
            hdr = []
        outhdr = [f for f in hdr if f != self.field]
        outhdr.insert(self.index, self.field)
        yield tuple(outhdr)
        outflds = list(map(str, outhdr))
        indices = asindices(hdr, outflds)
        transform = rowgetter(*indices)
        for row in it:
            try:
                yield transform(row)
            except IndexError:
                yield tuple((row[i] if i < len(row) else self.missing for i in indices))