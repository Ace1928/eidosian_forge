from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.maps import fieldmap, rowmap, rowmapmany
from functools import partial
def test_rowmap_empty():
    table = (('id', 'sex', 'age', 'height', 'weight'),)

    def rowmapper(row):
        transmf = {'male': 'M', 'female': 'F'}
        return [row[0], transmf[row[1]] if row[1] in transmf else row[1], row[2] * 12, row[4] / row[3] ** 2]
    actual = rowmap(table, rowmapper, header=['subject_id', 'gender', 'age_months', 'bmi'])
    expect = (('subject_id', 'gender', 'age_months', 'bmi'),)
    ieq(expect, actual)