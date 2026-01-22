from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import next
from petl.comparison import Comparable
from petl.util.base import header, Table
from petl.transform.sorts import sort
from petl.transform.basics import cut
class IntersectionView(Table):

    def __init__(self, a, b, presorted=False, buffersize=None, tempdir=None, cache=True):
        if presorted:
            self.a = a
            self.b = b
        else:
            self.a = sort(a, buffersize=buffersize, tempdir=tempdir, cache=cache)
            self.b = sort(b, buffersize=buffersize, tempdir=tempdir, cache=cache)

    def __iter__(self):
        return iterintersection(self.a, self.b)