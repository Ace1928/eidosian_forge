from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def test_unjoin_explicit_key_5():
    table6 = (('ColA', 'ColB', 'ColC'), ('A', 1, 'apple'), ('B', 1, 'apple'), ('C', 2, 'orange'), ('D', 3, 'lemon'), ('E', 3, 'lemon'))
    expect_left = (('ColA', 'ColB'), ('A', 1), ('B', 1), ('C', 2), ('D', 3), ('E', 3))
    expect_right = (('ColB', 'ColC'), (1, 'apple'), (2, 'orange'), (3, 'lemon'))
    left, right = unjoin(table6, 'ColC', key='ColB')
    ieq(expect_left, left)
    ieq(expect_left, left)
    ieq(expect_right, right)
    ieq(expect_right, right)