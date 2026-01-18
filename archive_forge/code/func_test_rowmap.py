from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.maps import fieldmap, rowmap, rowmapmany
from functools import partial
def test_rowmap():
    table = (('id', 'sex', 'age', 'height', 'weight'), (1, 'male', 16, 1.45, 62.0), (2, 'female', 19, 1.34, 55.4), (3, 'female', 17, 1.78, 74.4), (4, 'male', 21, 1.33, 45.2), (5, '-', 25, 1.65, 51.9))

    def rowmapper(row):
        transmf = {'male': 'M', 'female': 'F'}
        return [row[0], transmf[row[1]] if row[1] in transmf else row[1], row[2] * 12, row[4] / row[3] ** 2]
    actual = rowmap(table, rowmapper, header=['subject_id', 'gender', 'age_months', 'bmi'])
    expect = (('subject_id', 'gender', 'age_months', 'bmi'), (1, 'M', 16 * 12, 62.0 / 1.45 ** 2), (2, 'F', 19 * 12, 55.4 / 1.34 ** 2), (3, 'F', 17 * 12, 74.4 / 1.78 ** 2), (4, 'M', 21 * 12, 45.2 / 1.33 ** 2), (5, '-', 25 * 12, 51.9 / 1.65 ** 2))
    ieq(expect, actual)
    ieq(expect, actual)
    table2 = (('id', 'sex', 'age', 'height', 'weight'), (1, 'male', 16, 1.45, 62.0), (2, 'female', 19, 1.34, 55.4), (3, 'female', 17, 1.78, 74.4), (4, 'male', 21, 1.33, 45.2), (5, '-', 25, 1.65))
    expect = (('subject_id', 'gender', 'age_months', 'bmi'), (1, 'M', 16 * 12, 62.0 / 1.45 ** 2), (2, 'F', 19 * 12, 55.4 / 1.34 ** 2), (3, 'F', 17 * 12, 74.4 / 1.78 ** 2), (4, 'M', 21 * 12, 45.2 / 1.33 ** 2))
    actual = rowmap(table2, rowmapper, header=['subject_id', 'gender', 'age_months', 'bmi'])
    ieq(expect, actual)