from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import FieldSelectionError
from petl.util.base import Table, asindices, rowgetter
class PrefixHeaderView(Table):

    def __init__(self, table, prefix):
        self.table = table
        self.prefix = prefix

    def __iter__(self):
        it = iter(self.table)
        try:
            hdr = next(it)
        except StopIteration:
            return
        outhdr = tuple((text_type(self.prefix) + text_type(f) for f in hdr))
        yield outhdr
        for row in it:
            yield row