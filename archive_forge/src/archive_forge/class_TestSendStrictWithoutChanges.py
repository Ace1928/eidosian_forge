from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
class TestSendStrictWithoutChanges(tests.TestCaseWithTransport, TestSendStrictMixin):

    def setUp(self):
        super().setUp()
        self.parent, self.local = self.make_parent_and_local_branches()

    def test_send_without_workingtree(self):
        ControlDir.open('local').destroy_workingtree()
        self.assertSendSucceeds([])

    def test_send_default(self):
        self.assertSendSucceeds([])

    def test_send_strict(self):
        self.assertSendSucceeds(['--strict'])

    def test_send_no_strict(self):
        self.assertSendSucceeds(['--no-strict'])

    def test_send_config_var_strict(self):
        self.set_config_send_strict('true')
        self.assertSendSucceeds([])

    def test_send_config_var_no_strict(self):
        self.set_config_send_strict('false')
        self.assertSendSucceeds([])