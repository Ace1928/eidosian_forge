from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_send_strict_command_line_override_config(self):
    self.set_config_send_strict('false')
    self.assertSendSucceeds([])
    self.assertSendFails(['--strict'])