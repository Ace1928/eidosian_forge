from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_matche_float_upper_mixed_range_with_op_rangein_lt(self):
    self._do_specs_matcher_test(matches=False, value='20.5', req='<range-in> [ 10 20.5 )')