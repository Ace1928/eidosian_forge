from __future__ import absolute_import, division, print_function
import sys
import math
from collections import OrderedDict
from datetime import datetime, date, time
from decimal import Decimal
from petl.compat import izip_longest, text_type, string_types, PY3
from petl.io.sources import read_source_from_arg, write_source_from_arg
from petl.transform.headers import skip, setheader
from petl.util.base import Table, dicts, fieldnames, iterpeek, wrap
def _ordered_dict_iterator(table):
    it = iter(table)
    hdr = next(it)
    flds = [text_type(f) for f in hdr]
    for row in it:
        items = list()
        for i, f in enumerate(flds):
            try:
                v = row[i]
            except IndexError:
                v = None
            items.append((f, v))
        yield OrderedDict(items)