from __future__ import absolute_import, print_function, division
import os
import heapq
from tempfile import NamedTemporaryFile
import itertools
import logging
from collections import namedtuple
import operator
from petl.compat import pickle, next, text_type
import petl.config as config
from petl.comparison import comparable_itemgetter
from petl.util.base import Table, asindices
def _standardisedata(it, hdr, ofs):
    flds = list(map(text_type, hdr))
    for _row in it:
        try:
            yield tuple((_row[flds.index(fo)] if fo in flds else missing for fo in ofs))
        except IndexError:
            outrow = [missing] * len(ofs)
            for i, fi in enumerate(flds):
                try:
                    outrow[ofs.index(fi)] = _row[i]
                except IndexError:
                    pass
            yield tuple(outrow)