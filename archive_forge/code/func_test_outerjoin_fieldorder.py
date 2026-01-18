from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def test_outerjoin_fieldorder():
    table1 = (('colour', 'id'), ('blue', 1), ('red', 2), ('purple', 3))
    table2 = (('id', 'shape'), (1, 'circle'), (3, 'square'), (4, 'ellipse'))
    table3 = outerjoin(table1, table2, key='id')
    expect3 = (('colour', 'id', 'shape'), ('blue', 1, 'circle'), ('red', 2, None), ('purple', 3, 'square'), (None, 4, 'ellipse'))
    ieq(expect3, table3)
    ieq(expect3, table3)