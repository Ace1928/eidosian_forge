from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class SuicideDelete(TestBase):
    """Invalid weave which tries to add and delete simultaneously."""

    def runTest(self):
        k = Weave()
        k._parents = [()]
        k._weave = [(b'{', 0), b'first line', (b'[', 0), b'deleted in 0', (b']', 0), (b'}', 0)]
        return
        self.assertRaises(WeaveFormatError, k.get_lines, 0)