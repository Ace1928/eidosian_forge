from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import string_types, maketrans
from petl.util.base import values, Table, data, wrap
class ValueCountsView(Table):

    def __init__(self, table, field, missing=None):
        self.table = table
        self.field = field
        self.missing = missing

    def __iter__(self):
        if isinstance(self.field, (tuple, list)):
            outhdr = tuple(self.field) + ('count', 'frequency')
        else:
            outhdr = (self.field, 'count', 'frequency')
        yield outhdr
        counter = valuecounter(self.table, *self.field, missing=self.missing)
        counts = counter.most_common()
        total = sum((c[1] for c in counts))
        if len(self.field) > 1:
            for c in counts:
                yield (tuple(c[0]) + (c[1], float(c[1]) / total))
        else:
            for c in counts:
                yield (c[0], c[1], float(c[1]) / total)