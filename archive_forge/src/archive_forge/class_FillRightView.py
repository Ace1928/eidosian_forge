from __future__ import absolute_import, print_function, division
from petl.compat import next
from petl.util.base import Table, asindices
class FillRightView(Table):

    def __init__(self, table, missing=None):
        self.table = table
        self.missing = missing

    def __iter__(self):
        return iterfillright(self.table, self.missing)