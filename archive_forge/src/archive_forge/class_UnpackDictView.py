from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.util.base import Table
class UnpackDictView(Table):

    def __init__(self, table, field, keys=None, includeoriginal=False, samplesize=1000, missing=None):
        self.table = table
        self.field = field
        self.keys = keys
        self.includeoriginal = includeoriginal
        self.samplesize = samplesize
        self.missing = missing

    def __iter__(self):
        return iterunpackdict(self.table, self.field, self.keys, self.includeoriginal, self.samplesize, self.missing)