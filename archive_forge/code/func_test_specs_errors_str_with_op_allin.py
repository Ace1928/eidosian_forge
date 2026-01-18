from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_errors_str_with_op_allin(self):
    self.assertRaises(TypeError, specs_matcher.match, value='aes', req='<all-in> aes')