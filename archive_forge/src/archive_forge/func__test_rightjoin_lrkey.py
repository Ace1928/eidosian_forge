from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_rightjoin_lrkey(rightjoin_impl):
    table1 = (('id', 'colour'), (1, 'blue'), (2, 'red'), (3, 'purple'))
    table2 = (('identifier', 'shape'), (0, 'triangle'), (1, 'circle'), (3, 'square'), (4, 'ellipse'), (5, 'pentagon'))
    table3 = rightjoin_impl(table1, table2, lkey='id', rkey='identifier')
    expect3 = (('id', 'colour', 'shape'), (0, None, 'triangle'), (1, 'blue', 'circle'), (3, 'purple', 'square'), (4, None, 'ellipse'), (5, None, 'pentagon'))
    ieq(expect3, table3)