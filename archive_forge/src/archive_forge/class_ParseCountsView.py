from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import string_types, maketrans
from petl.util.base import values, Table, data, wrap
class ParseCountsView(Table):

    def __init__(self, table, field, parsers=(('int', int), ('float', float))):
        self.table = table
        self.field = field
        if isinstance(parsers, (list, tuple)):
            parsers = dict(parsers)
        self.parsers = parsers

    def __iter__(self):
        counter, errors = parsecounter(self.table, self.field, self.parsers)
        yield ('type', 'count', 'errors')
        for item, n in counter.most_common():
            yield (item, n, errors[item])