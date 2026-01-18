from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def test_outerjoin_novaluefield():
    table1 = (('id', 'colour'), (0, 'black'), (1, 'blue'), (2, 'red'), (3, 'purple'), (5, 'yellow'), (7, 'white'))
    table2 = (('id', 'shape'), (1, 'circle'), (3, 'square'), (4, 'ellipse'))
    expect = (('id', 'colour', 'shape'), (0, 'black', None), (1, 'blue', 'circle'), (2, 'red', None), (3, 'purple', 'square'), (4, None, 'ellipse'), (5, 'yellow', None), (7, 'white', None))
    actual = outerjoin(table1, table2, key='id')
    ieq(expect, actual)
    actual = outerjoin(cut(table1, 'id'), table2, key='id')
    ieq(cut(expect, 'id', 'shape'), actual)
    actual = outerjoin(table1, cut(table2, 'id'), key='id')
    ieq(cut(expect, 'id', 'colour'), actual)
    actual = outerjoin(cut(table1, 'id'), cut(table2, 'id'), key='id')
    ieq(cut(expect, 'id'), actual)