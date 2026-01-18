from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_match_int_leading_zero(self):
    self._do_specs_matcher_test(value='01', req='== 1', matches=True)