from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_send_default(self):
    self.assertSendSucceeds([], with_warning=True)