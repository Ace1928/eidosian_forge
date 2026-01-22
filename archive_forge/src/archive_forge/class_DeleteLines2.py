from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class DeleteLines2(TestBase):
    """Test recording revisions that delete lines.

    This relies on the weave having a way to represent lines knocked
    out by a later revision."""

    def runTest(self):
        k = Weave()
        k.add_lines(b'text0', [], [b'line the first', b'line 2', b'line 3', b'fine'])
        self.assertEqual(len(k.get_lines(0)), 4)
        k.add_lines(b'text1', [b'text0'], [b'line the first', b'fine'])
        self.assertEqual(k.get_lines(1), [b'line the first', b'fine'])
        self.assertEqual(k.annotate(b'text1'), [(b'text0', b'line the first'), (b'text0', b'fine')])