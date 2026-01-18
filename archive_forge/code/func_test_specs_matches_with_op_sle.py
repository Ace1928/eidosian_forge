from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_matches_with_op_sle(self):
    self._do_specs_matcher_test(value='1000', req='s<= 1234', matches=True)