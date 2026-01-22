from __future__ import absolute_import, print_function, division
import itertools
import collections
import operator
from petl.compat import next, text_type
from petl.comparison import comparable_itemgetter
from petl.util.base import Table, rowgetter, values, itervalues, \
from petl.transform.sorts import sort
class MeltView(Table):

    def __init__(self, source, key=None, variables=None, variablefield='variable', valuefield='value'):
        self.source = source
        self.key = key
        self.variables = variables
        self.variablefield = variablefield
        self.valuefield = valuefield

    def __iter__(self):
        return itermelt(self.source, self.key, self.variables, self.variablefield, self.valuefield)