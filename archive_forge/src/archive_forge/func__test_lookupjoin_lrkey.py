from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_lookupjoin_lrkey(lookupjoin_impl):
    table1 = (('id', 'color', 'cost'), (1, 'blue', 12), (2, 'red', 8), (3, 'purple', 4))
    table2 = (('identifier', 'shape', 'size'), (1, 'circle', 'big'), (2, 'square', 'tiny'), (3, 'ellipse', 'small'))
    actual = lookupjoin_impl(table1, table2, lkey='id', rkey='identifier')
    expect = (('id', 'color', 'cost', 'shape', 'size'), (1, 'blue', 12, 'circle', 'big'), (2, 'red', 8, 'square', 'tiny'), (3, 'purple', 4, 'ellipse', 'small'))
    ieq(expect, actual)