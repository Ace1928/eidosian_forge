from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def _search_tree(tree, start, stop, include_stop):
    if stop is None:
        if include_stop:
            stop = start + 1
            start -= 1
            args = (start, stop)
        else:
            args = (start,)
    else:
        if include_stop:
            stop += 1
            start -= 1
        args = (start, stop)
    if len(args) == 2:
        results = sorted(tree.overlap(*args))
    else:
        results = sorted(tree.at(*args))
    return results