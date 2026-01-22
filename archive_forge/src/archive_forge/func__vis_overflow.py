from __future__ import absolute_import, print_function, division
import locale
from itertools import islice
from collections import defaultdict
from petl.compat import numeric_types, text_type
from petl import config
from petl.util.base import Table
from petl.io.sources import MemorySource
from petl.io.html import tohtml
def _vis_overflow(table, limit):
    overflow = False
    if limit:
        table = list(islice(table, 0, limit + 2))
        if len(table) > limit + 1:
            overflow = True
            table = table[:-1]
    return (table, overflow)