from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_antijoin_basics(antijoin_impl):
    table1 = (('id', 'colour'), (0, 'black'), (1, 'blue'), (2, 'red'), (4, 'yellow'), (5, 'white'))
    table2 = (('id', 'shape'), (1, 'circle'), (3, 'square'))
    table3 = antijoin_impl(table1, table2, key='id')
    expect3 = (('id', 'colour'), (0, 'black'), (2, 'red'), (4, 'yellow'), (5, 'white'))
    ieq(expect3, table3)
    table4 = antijoin_impl(table1, table2)
    expect4 = expect3
    ieq(expect4, table4)