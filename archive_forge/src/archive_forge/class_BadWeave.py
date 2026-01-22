from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class BadWeave(TestBase):
    """Test that we trap an insert which should not occur."""

    def runTest(self):
        k = Weave()
        k._parents = [frozenset()]
        k._weave = [b'bad line', (b'{', 0), b'foo {', (b'{', 1), b'  added in version 1', (b'{', 2), b'  added in v2', (b'}', 2), b'  also from v1', (b'}', 1), b'}', (b'}', 0)]
        return
        self.assertRaises(WeaveFormatError, k.get, 0)