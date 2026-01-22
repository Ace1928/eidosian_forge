from __future__ import absolute_import, print_function, division
import operator
from petl.compat import text_type
from petl.util.base import Table, asindices, itervalues
from petl.transform.sorts import sort
class ConflictsView(Table):

    def __init__(self, source, key, missing=None, exclude=None, include=None, presorted=False, buffersize=None, tempdir=None, cache=True):
        if presorted:
            self.source = source
        else:
            self.source = sort(source, key, buffersize=buffersize, tempdir=tempdir, cache=cache)
        self.key = key
        self.missing = missing
        self.exclude = exclude
        self.include = include

    def __iter__(self):
        return iterconflicts(self.source, self.key, self.missing, self.exclude, self.include)