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
def test_sort_4():
    table = (('foo', 'bar'), ('C', 2), ('A', 9), ('A', 6), ('F', 1), ('D', 10))
    result = sort(table, 'bar')
    expectation = (('foo', 'bar'), ('F', 1), ('C', 2), ('A', 6), ('A', 9), ('D', 10))
    ieq(expectation, result)