from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import FieldSelectionError
from petl.util.base import Table, asindices, rowgetter
class SortHeaderView(Table):

    def __init__(self, table, reverse, missing):
        self.table = table
        self.reverse = reverse
        self.missing = missing

    def __iter__(self):
        it = iter(self.table)
        try:
            hdr = next(it)
        except StopIteration:
            return
        shdr = sorted(hdr)
        indices = asindices(hdr, shdr)
        transform = rowgetter(*indices)
        yield tuple(shdr)
        missing = self.missing
        for row in it:
            try:
                yield transform(row)
            except IndexError:
                yield tuple((row[i] if i < len(row) else missing for i in indices))