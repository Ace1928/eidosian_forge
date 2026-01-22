from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
class IntervalJoinView(Table):

    def __init__(self, left, right, lstart='start', lstop='stop', rstart='start', rstop='stop', lkey=None, rkey=None, include_stop=False, lprefix=None, rprefix=None):
        self.left = left
        self.lstart = lstart
        self.lstop = lstop
        self.lkey = lkey
        self.right = right
        self.rstart = rstart
        self.rstop = rstop
        self.rkey = rkey
        self.include_stop = include_stop
        self.lprefix = lprefix
        self.rprefix = rprefix

    def __iter__(self):
        return iterintervaljoin(left=self.left, right=self.right, lstart=self.lstart, lstop=self.lstop, rstart=self.rstart, rstop=self.rstop, lkey=self.lkey, rkey=self.rkey, include_stop=self.include_stop, missing=None, lprefix=self.lprefix, rprefix=self.rprefix, leftouter=False)