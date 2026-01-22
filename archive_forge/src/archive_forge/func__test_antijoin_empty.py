from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_antijoin_empty(antijoin_impl):
    table1 = (('id', 'colour'), (0, 'black'), (1, 'blue'), (2, 'red'), (4, 'yellow'), (5, 'white'))
    table2 = (('id', 'shape'),)
    actual = antijoin_impl(table1, table2, key='id')
    expect = table1
    ieq(expect, actual)