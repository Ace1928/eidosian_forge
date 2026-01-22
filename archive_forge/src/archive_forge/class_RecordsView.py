from __future__ import absolute_import, print_function, division
import re
from itertools import islice, chain, cycle, product,\
import operator
from collections import Counter, namedtuple, OrderedDict
from itertools import compress, combinations_with_replacement
from petl.compat import imap, izip, izip_longest, ifilter, ifilterfalse, \
from petl.errors import FieldSelectionError
from petl.comparison import comparable_itemgetter
class RecordsView(IterContainer):

    def __init__(self, table, *sliceargs, **kwargs):
        self.table = table
        self.sliceargs = sliceargs
        self.kwargs = kwargs

    def __iter__(self):
        return iterrecords(self.table, *self.sliceargs, **self.kwargs)

    def __repr__(self):
        vreprs = list(map(repr, islice(self, 6)))
        r = '\n'.join(vreprs[:5])
        if len(vreprs) > 5:
            r += '\n...'
        return r