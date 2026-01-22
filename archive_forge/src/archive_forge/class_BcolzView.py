from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import string_types, text_type
from petl.util.base import Table, iterpeek
from petl.io.numpy import construct_dtype
class BcolzView(Table):

    def __init__(self, source, expression=None, outcols=None, limit=None, skip=0):
        self.source = source
        self.expression = expression
        self.outcols = outcols
        self.limit = limit
        self.skip = skip

    def __iter__(self):
        if isinstance(self.source, string_types):
            import bcolz
            ctbl = bcolz.open(self.source, mode='r')
        else:
            ctbl = self.source
        if self.outcols is None:
            header = tuple(ctbl.names)
        else:
            header = tuple(self.outcols)
            assert all((h in ctbl.names for h in header)), 'invalid outcols'
        yield header
        if self.expression is None:
            it = ctbl.iter(outcols=self.outcols, skip=self.skip, limit=self.limit)
        else:
            it = ctbl.where(self.expression, outcols=self.outcols, skip=self.skip, limit=self.limit)
        for row in it:
            yield row