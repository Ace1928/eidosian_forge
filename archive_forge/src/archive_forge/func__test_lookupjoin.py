from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_lookupjoin(lookupjoin_impl):
    _test_lookupjoin_1(lookupjoin_impl)
    _test_lookupjoin_2(lookupjoin_impl)
    _test_lookupjoin_prefix(lookupjoin_impl)
    _test_lookupjoin_lrkey(lookupjoin_impl)
    _test_lookupjoin_novaluefield(lookupjoin_impl)