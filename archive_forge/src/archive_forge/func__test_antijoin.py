from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_antijoin(antijoin_impl):
    _test_antijoin_basics(antijoin_impl)
    _test_antijoin_empty(antijoin_impl)
    _test_antijoin_novaluefield(antijoin_impl)
    _test_antijoin_lrkey(antijoin_impl)