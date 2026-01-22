from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
class TestBundleStrictWithoutChanges(TestSendStrictWithoutChanges):
    _default_command = ['bundle-revisions', '../parent']