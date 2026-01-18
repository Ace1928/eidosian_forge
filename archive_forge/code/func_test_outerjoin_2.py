from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def test_outerjoin_2():
    table1 = (('id', 'colour'), (1, 'blue'), (2, 'red'), (3, 'purple'))
    table2 = (('id', 'shape'), (0, 'pentagon'), (1, 'circle'), (3, 'square'), (4, 'ellipse'), (5, 'triangle'))
    table3 = outerjoin(table1, table2, key='id')
    expect3 = (('id', 'colour', 'shape'), (0, None, 'pentagon'), (1, 'blue', 'circle'), (2, 'red', None), (3, 'purple', 'square'), (4, None, 'ellipse'), (5, None, 'triangle'))
    ieq(expect3, table3)
    ieq(expect3, table3)
    table4 = outerjoin(table1, table2)
    expect4 = expect3
    ieq(expect4, table4)