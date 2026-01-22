from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_join(join_impl):
    _test_join_basic(join_impl)
    _test_join_compound_keys(join_impl)
    _test_join_string_key(join_impl)
    _test_join_empty(join_impl)
    _test_join_novaluefield(join_impl)
    _test_join_prefix(join_impl)
    _test_join_lrkey(join_impl)
    _test_join_multiple(join_impl)