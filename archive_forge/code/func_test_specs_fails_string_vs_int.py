from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_string_vs_int(self):
    self._do_specs_matcher_test(value='01', req='1', matches=False)