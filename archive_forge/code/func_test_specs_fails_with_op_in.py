from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_with_op_in(self):
    self._do_specs_matcher_test(value='12310321', req='<in> 11', matches=False)