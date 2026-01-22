from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_leftjoin_prefix(leftjoin_impl):
    table1 = (('id', 'colour'), (1, 'blue'), (2, 'red'), (3, 'purple'), (5, 'yellow'), (7, 'orange'))
    table2 = (('id', 'shape'), (1, 'circle'), (3, 'square'), (4, 'ellipse'))
    table3 = leftjoin_impl(table1, table2, key='id', lprefix='l_', rprefix='r_')
    expect3 = (('l_id', 'l_colour', 'r_shape'), (1, 'blue', 'circle'), (2, 'red', None), (3, 'purple', 'square'), (5, 'yellow', None), (7, 'orange', None))
    ieq(expect3, table3)