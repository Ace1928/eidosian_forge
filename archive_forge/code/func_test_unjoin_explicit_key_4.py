from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def test_unjoin_explicit_key_4():
    table5 = (('Restaurant', 'Pizza Variety', 'Delivery Area'), ('A1 Pizza', 'Thick Crust', 'Springfield'), ('A1 Pizza', 'Thick Crust', 'Shelbyville'), ('A1 Pizza', 'Thick Crust', 'Capital City'), ('A1 Pizza', 'Stuffed Crust', 'Springfield'), ('A1 Pizza', 'Stuffed Crust', 'Shelbyville'), ('A1 Pizza', 'Stuffed Crust', 'Capital City'), ('Elite Pizza', 'Thin Crust', 'Capital City'), ('Elite Pizza', 'Stuffed Crust', 'Capital City'), ("Vincenzo's Pizza", 'Thick Crust', 'Springfield'), ("Vincenzo's Pizza", 'Thick Crust', 'Shelbyville'), ("Vincenzo's Pizza", 'Thin Crust', 'Springfield'), ("Vincenzo's Pizza", 'Thin Crust', 'Shelbyville'))
    expect_left = (('Restaurant', 'Pizza Variety'), ('A1 Pizza', 'Stuffed Crust'), ('A1 Pizza', 'Thick Crust'), ('Elite Pizza', 'Stuffed Crust'), ('Elite Pizza', 'Thin Crust'), ("Vincenzo's Pizza", 'Thick Crust'), ("Vincenzo's Pizza", 'Thin Crust'))
    expect_right = (('Restaurant', 'Delivery Area'), ('A1 Pizza', 'Capital City'), ('A1 Pizza', 'Shelbyville'), ('A1 Pizza', 'Springfield'), ('Elite Pizza', 'Capital City'), ("Vincenzo's Pizza", 'Shelbyville'), ("Vincenzo's Pizza", 'Springfield'))
    left, right = unjoin(table5, 'Delivery Area', key='Restaurant')
    ieq(expect_left, left)
    ieq(expect_left, left)
    ieq(expect_right, right)
    ieq(expect_right, right)