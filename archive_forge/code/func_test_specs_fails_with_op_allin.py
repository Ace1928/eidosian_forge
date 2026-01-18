from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_with_op_allin(self):
    self._do_specs_matcher_test(value=str(['aes', 'mmx', 'aux']), req='<all-in>  txt', matches=False)