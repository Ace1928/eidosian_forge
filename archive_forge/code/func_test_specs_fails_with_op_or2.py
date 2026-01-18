from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_with_op_or2(self):
    self._do_specs_matcher_test(value='13', req='<or> 11 <or> 12 <or>', matches=False)