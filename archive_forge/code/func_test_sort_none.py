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
def test_sort_none():
    table = (('foo', 'bar'), ('C', 2), ('A', 9), ('A', None), ('F', 1), ('D', 10))
    result = sort(table, 'bar')
    print(list(result))
    expectation = (('foo', 'bar'), ('A', None), ('F', 1), ('C', 2), ('A', 9), ('D', 10))
    ieq(expectation, result)
    dt = datetime.now().replace
    table = (('foo', 'bar'), ('C', dt(hour=5)), ('A', dt(hour=1)), ('A', None), ('F', dt(hour=9)), ('D', dt(hour=17)))
    result = sort(table, 'bar')
    expectation = (('foo', 'bar'), ('A', None), ('A', dt(hour=1)), ('C', dt(hour=5)), ('F', dt(hour=9)), ('D', dt(hour=17)))
    ieq(expectation, result)