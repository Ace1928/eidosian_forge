from __future__ import absolute_import, print_function, division
import os
import gc
import logging
from datetime import datetime
import platform
import pytest
from petl.compat import next
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq, eq_
from petl.util import nrows
from petl.transform.basics import cat
from petl.transform.sorts import sort, mergesort, issorted
def test_sort_buffered_cleanup():
    table = (('foo', 'bar'), ('C', 2), ('A', 9), ('A', 6), ('F', 1), ('D', 10))
    result = sort(table, 'bar', buffersize=2)
    debug('initially filecache should be empty')
    eq_(None, result._filecache)
    debug('pull rows through, should populate file cache')
    eq_(5, nrows(result))
    eq_(3, len(result._filecache))
    debug('check all files exist')
    filenames = _get_names(result._filecache)
    for fn in filenames:
        assert os.path.exists(fn), fn
    debug('delete object and garbage collect')
    del result
    gc.collect()
    debug('check all files have been deleted')
    for fn in filenames:
        assert not os.path.exists(fn), fn