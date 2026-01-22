from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_leftjoin_3(leftjoin_impl):
    table1 = (('id', 'colour'), (1, 'blue'), (2, 'red'), (3, 'purple'))
    table2 = (('id', 'shape'), (1, 'circle'), (3, 'square'), (4, 'ellipse'), (5, 'triangle'))
    table3 = leftjoin_impl(table1, table2, key='id')
    expect3 = (('id', 'colour', 'shape'), (1, 'blue', 'circle'), (2, 'red', None), (3, 'purple', 'square'))
    ieq(expect3, table3)
    ieq(expect3, table3)
    table4 = leftjoin_impl(table1, table2)
    expect4 = expect3
    ieq(expect4, table4)