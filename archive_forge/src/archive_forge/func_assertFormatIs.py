from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def assertFormatIs(self, fmt_string, md):
    self.assertEqual(fmt_string, md.get_raw_bundle().splitlines()[0])