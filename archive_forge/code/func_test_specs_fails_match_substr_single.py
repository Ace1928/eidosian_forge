from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_match_substr_single(self):
    self._do_specs_matcher_test(value=str(['X_X']), req='<all-in> _', matches=False)