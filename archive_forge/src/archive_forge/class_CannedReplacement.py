from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class CannedReplacement(TestBase):
    """Unpack canned weave with deleted lines."""

    def runTest(self):
        k = Weave()
        k._parents = [frozenset(), frozenset([0])]
        k._weave = [(b'{', 0), b'first line', (b'[', 1), b'line to be deleted', (b']', 1), (b'{', 1), b'replacement line', (b'}', 1), b'last line', (b'}', 0)]
        k._sha1s = [sha_string(b'first lineline to be deletedlast line'), sha_string(b'first linereplacement linelast line')]
        self.assertEqual(k.get_lines(0), [b'first line', b'line to be deleted', b'last line'])
        self.assertEqual(k.get_lines(1), [b'first line', b'replacement line', b'last line'])