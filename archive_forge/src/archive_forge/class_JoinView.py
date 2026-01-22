from __future__ import absolute_import, print_function, division
import itertools
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.comparison import comparable_itemgetter, Comparable
from petl.util.base import Table, asindices, rowgetter, rowgroupby, \
from petl.transform.sorts import sort
from petl.transform.basics import cut, cutout
from petl.transform.dedup import distinct
class JoinView(Table):

    def __init__(self, left, right, lkey, rkey, presorted=False, leftouter=False, rightouter=False, missing=None, buffersize=None, tempdir=None, cache=True, lprefix=None, rprefix=None):
        self.lkey = lkey
        self.rkey = rkey
        if presorted:
            self.left = left
            self.right = right
        else:
            self.left = sort(left, lkey, buffersize=buffersize, tempdir=tempdir, cache=cache)
            self.right = sort(right, rkey, buffersize=buffersize, tempdir=tempdir, cache=cache)
        self.leftouter = leftouter
        self.rightouter = rightouter
        self.missing = missing
        self.lprefix = lprefix
        self.rprefix = rprefix

    def __iter__(self):
        return iterjoin(self.left, self.right, self.lkey, self.rkey, leftouter=self.leftouter, rightouter=self.rightouter, missing=self.missing, lprefix=self.lprefix, rprefix=self.rprefix)