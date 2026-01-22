from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import next
from petl.comparison import Comparable
from petl.util.base import header, Table
from petl.transform.sorts import sort
from petl.transform.basics import cut
class HashComplementView(Table):

    def __init__(self, a, b, strict=False):
        self.a = a
        self.b = b
        self.strict = strict

    def __iter__(self):
        return iterhashcomplement(self.a, self.b, self.strict)