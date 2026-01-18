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
def test_sort_buffered():
    table = (('foo', 'bar'), ('C', 2), ('A', 9), ('A', 6), ('F', 1), ('D', 10))
    expectation = (('foo', 'bar'), ('F', 1), ('C', 2), ('A', 6), ('A', 9), ('D', 10))
    result = sort(table, 'bar')
    ieq(expectation, result)
    result = sort(table, 'bar', buffersize=2)
    ieq(expectation, result)
    expectation = (('foo', 'bar'), ('D', 10), ('A', 9), ('A', 6), ('C', 2), ('F', 1))
    result = sort(table, 'bar', reverse=True)
    ieq(expectation, result)
    result = sort(table, 'bar', reverse=True, buffersize=2)
    ieq(expectation, result)
    expectation = (('foo', 'bar'), ('F', 1), ('D', 10), ('C', 2), ('A', 9), ('A', 6))
    result = sort(table, reverse=True)
    ieq(expectation, result)
    result = sort(table, reverse=True, buffersize=2)
    ieq(expectation, result)