from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_matches_float_lower_float_range_with_op_rangein_ge(self):
    self._do_specs_matcher_test(matches=True, value='10.1', req='<range-in> [ 10.1 20 ]')