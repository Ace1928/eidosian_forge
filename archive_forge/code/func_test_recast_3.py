from __future__ import absolute_import, print_function, division
from datetime import datetime
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.reshape import melt, recast, transpose, pivot, flatten, \
from petl.transform.regex import split, capture
def test_recast_3():
    table = (('id', 'time', 'variable', 'value'), (1, 11, 'weight', 66.4), (1, 14, 'weight', 55.2), (2, 12, 'weight', 53.2), (2, 16, 'weight', 43.3), (3, 12, 'weight', 34.5), (3, 17, 'weight', 49.4))
    expectation = (('id', 'time', 'weight'), (1, 11, 66.4), (1, 14, 55.2), (2, 12, 53.2), (2, 16, 43.3), (3, 12, 34.5), (3, 17, 49.4))
    result = recast(table)
    ieq(expectation, result)
    expectation = (('id', 'weight'), (1, [66.4, 55.2]), (2, [53.2, 43.3]), (3, [34.5, 49.4]))
    result = recast(table, key='id')
    ieq(expectation, result)
    expectation = (('id', 'weight'), (1, 66.4), (2, 53.2), (3, 49.4))
    result = recast(table, key='id', reducers={'weight': max})
    ieq(expectation, result)
    expectation = (('id', 'weight'), (1, 55.2), (2, 43.3), (3, 34.5))
    result = recast(table, key='id', reducers={'weight': min})
    ieq(expectation, result)
    expectation = (('id', 'weight'), (1, 60.8), (2, 48.25), (3, 41.95))

    def mean(values):
        return float(sum(values)) / len(values)

    def meanf(precision):

        def f(values):
            v = mean(values)
            v = round(v, precision)
            return v
        return f
    result = recast(table, key='id', reducers={'weight': meanf(precision=2)})
    ieq(expectation, result)