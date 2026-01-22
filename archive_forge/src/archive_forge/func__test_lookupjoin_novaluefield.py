from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_lookupjoin_novaluefield(lookupjoin_impl):
    table1 = (('id', 'color', 'cost'), (1, 'blue', 12), (2, 'red', 8), (3, 'purple', 4))
    table2 = (('id', 'shape', 'size'), (1, 'circle', 'big'), (2, 'square', 'tiny'), (3, 'ellipse', 'small'))
    expect = (('id', 'color', 'cost', 'shape', 'size'), (1, 'blue', 12, 'circle', 'big'), (2, 'red', 8, 'square', 'tiny'), (3, 'purple', 4, 'ellipse', 'small'))
    actual = lookupjoin_impl(table1, table2, key='id')
    ieq(expect, actual)
    actual = lookupjoin_impl(cut(table1, 'id'), table2, key='id')
    ieq(cut(expect, 'id', 'shape', 'size'), actual)
    actual = lookupjoin_impl(table1, cut(table2, 'id'), key='id')
    ieq(cut(expect, 'id', 'color', 'cost'), actual)
    actual = lookupjoin_impl(cut(table1, 'id'), cut(table2, 'id'), key='id')
    ieq(cut(expect, 'id'), actual)