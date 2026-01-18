from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_matches_only_one_with_op_rangein(self):
    self._do_specs_matcher_test(matches=True, value='10.1', req='<range-in> [ 10.1 10.1 ]')