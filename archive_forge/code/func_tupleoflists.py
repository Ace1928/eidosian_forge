from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from itertools import islice
from petl.compat import izip_longest, text_type, next
from petl.util.base import asindices, Table
def tupleoflists(tbl):
    return tuple((list(row) for row in tbl))