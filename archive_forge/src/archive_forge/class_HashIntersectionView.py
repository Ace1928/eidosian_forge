from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import next
from petl.comparison import Comparable
from petl.util.base import header, Table
from petl.transform.sorts import sort
from petl.transform.basics import cut
class HashIntersectionView(Table):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __iter__(self):
        return iterhashintersection(self.a, self.b)