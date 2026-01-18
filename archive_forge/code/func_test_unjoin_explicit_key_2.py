from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def test_unjoin_explicit_key_2():
    table3 = (('Employee', 'Skill', 'Current Work Location'), ('Jones', 'Typing', '114 Main Street'), ('Jones', 'Shorthand', '114 Main Street'), ('Jones', 'Whittling', '114 Main Street'), ('Bravo', 'Light Cleaning', '73 Industrial Way'), ('Ellis', 'Alchemy', '73 Industrial Way'), ('Ellis', 'Flying', '73 Industrial Way'), ('Harrison', 'Light Cleaning', '73 Industrial Way'))
    expect_left = (('Employee', 'Current Work Location'), ('Bravo', '73 Industrial Way'), ('Ellis', '73 Industrial Way'), ('Harrison', '73 Industrial Way'), ('Jones', '114 Main Street'))
    expect_right = (('Employee', 'Skill'), ('Bravo', 'Light Cleaning'), ('Ellis', 'Alchemy'), ('Ellis', 'Flying'), ('Harrison', 'Light Cleaning'), ('Jones', 'Shorthand'), ('Jones', 'Typing'), ('Jones', 'Whittling'))
    left, right = unjoin(table3, 'Skill', key='Employee')
    ieq(expect_left, left)
    ieq(expect_left, left)
    ieq(expect_right, right)
    ieq(expect_right, right)