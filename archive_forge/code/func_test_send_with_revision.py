from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_send_with_revision(self):
    self.assertSendSucceeds(['-r', 'revid:' + self.local.decode('utf-8')], revs=[self.local])