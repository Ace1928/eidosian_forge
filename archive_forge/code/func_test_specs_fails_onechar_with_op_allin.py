from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_onechar_with_op_allin(self):
    self.assertRaises(TypeError, specs_matcher.match, value=str(['aes', 'mmx', 'aux']), req='<all-in> e')