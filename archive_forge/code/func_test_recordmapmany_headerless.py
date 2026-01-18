from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.maps import fieldmap, rowmap, rowmapmany
from functools import partial
def test_recordmapmany_headerless():
    table = []

    def duplicate(rec):
        yield rec
        yield rec
    actual = rowmapmany(table, duplicate, header=['subject_id', 'variable'])
    expect = []
    ieq(expect, actual)
    ieq(expect, actual)