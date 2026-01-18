from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def upstream(prv, cur, nxt):
    if prv is None:
        return cur.bar
    else:
        return cur.bar + prv.baz