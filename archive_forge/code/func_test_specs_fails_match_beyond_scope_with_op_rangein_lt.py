from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_match_beyond_scope_with_op_rangein_lt(self):
    self._do_specs_matcher_test(matches=False, value='23', req='<range-in> [ 10 20 )')