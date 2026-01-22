from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, text_type
from petl.util.base import Table, asindices, rowgetter, iterpeek
from petl.util.lookups import lookup, lookupone
from petl.transform.joins import keys_from_args
class HashLeftJoinView(Table):

    def __init__(self, left, right, lkey, rkey, missing=None, cache=True, lprefix=None, rprefix=None):
        self.left = left
        self.right = right
        self.lkey = lkey
        self.rkey = rkey
        self.missing = missing
        self.cache = cache
        self.rlookup = None
        self.lprefix = lprefix
        self.rprefix = rprefix

    def __iter__(self):
        if not self.cache or self.rlookup is None:
            self.rlookup = lookup(self.right, self.rkey)
        return iterhashleftjoin(self.left, self.right, self.lkey, self.rkey, self.missing, self.rlookup, self.lprefix, self.rprefix)