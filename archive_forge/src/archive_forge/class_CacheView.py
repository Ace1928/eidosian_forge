from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from itertools import islice
from petl.compat import izip_longest, text_type, next
from petl.util.base import asindices, Table
class CacheView(Table):

    def __init__(self, inner, n=None):
        self.inner = inner
        self.n = n
        self.cache = list()
        self.cachecomplete = False

    def clearcache(self):
        self.cache = list()
        self.cachecomplete = False

    def __iter__(self):
        for row in self.cache:
            yield row
        if not self.cachecomplete:
            it = iter(self.inner)
            for row in islice(it, len(self.cache), None):
                if not self.n or len(self.cache) < self.n:
                    self.cache.append(row)
                yield row
            if not self.n or len(self.cache) < self.n:
                self.cachecomplete = True