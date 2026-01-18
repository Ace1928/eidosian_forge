from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def test_unjoin_implicit_key():
    table1 = (('foo', 'bar'), (1, 'apple'), (2, 'apple'), (3, 'orange'))
    expect_left = (('foo', 'bar_id'), (1, 1), (2, 1), (3, 2))
    expect_right = (('id', 'bar'), (1, 'apple'), (2, 'orange'))
    left, right = unjoin(table1, 'bar')
    ieq(expect_left, left)
    ieq(expect_left, left)
    ieq(expect_right, right)
    ieq(expect_right, right)