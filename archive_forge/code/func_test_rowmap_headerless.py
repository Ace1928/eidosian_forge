from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.maps import fieldmap, rowmap, rowmapmany
from functools import partial
def test_rowmap_headerless():
    table = []

    def rowmapper(row):
        return row
    actual = rowmap(table, rowmapper, header=['subject_id', 'gender'])
    expect = []
    ieq(expect, actual)