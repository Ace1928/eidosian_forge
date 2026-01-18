from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def test_outerjoin_empty():
    table1 = (('id', 'colour'), (0, 'black'), (1, 'blue'), (2, 'red'), (3, 'purple'), (5, 'yellow'), (7, 'white'))
    table2 = (('id', 'shape'),)
    table3 = outerjoin(table1, table2, key='id')
    expect3 = (('id', 'colour', 'shape'), (0, 'black', None), (1, 'blue', None), (2, 'red', None), (3, 'purple', None), (5, 'yellow', None), (7, 'white', None))
    ieq(expect3, table3)