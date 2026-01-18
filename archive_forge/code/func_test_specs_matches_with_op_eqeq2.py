from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_matches_with_op_eqeq2(self):
    self._do_specs_matcher_test(value='3.0', req='== 3', matches=True)