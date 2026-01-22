from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_rightjoin_empty(rightjoin_impl):
    table1 = (('id', 'colour'),)
    table2 = (('id', 'shape'), (0, 'triangle'), (1, 'circle'), (3, 'square'), (4, 'ellipse'), (5, 'pentagon'))
    table3 = rightjoin_impl(table1, table2, key='id')
    expect3 = (('id', 'colour', 'shape'), (0, None, 'triangle'), (1, None, 'circle'), (3, None, 'square'), (4, None, 'ellipse'), (5, None, 'pentagon'))
    ieq(expect3, table3)