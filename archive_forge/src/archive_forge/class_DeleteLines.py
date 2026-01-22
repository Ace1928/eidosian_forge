from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class DeleteLines(TestBase):
    """Deletion of lines from existing text.

    Try various texts all based on a common ancestor."""

    def runTest(self):
        k = Weave()
        base_text = [b'one', b'two', b'three', b'four']
        k.add_lines(b'text0', [], base_text)
        texts = [[b'one', b'two', b'three'], [b'two', b'three', b'four'], [b'one', b'four'], [b'one', b'two', b'three', b'four']]
        i = 1
        for t in texts:
            k.add_lines(b'text%d' % i, [b'text0'], t)
            i += 1
        self.log('final weave:')
        self.log('k._weave=' + pformat(k._weave))
        for i in range(len(texts)):
            self.assertEqual(k.get_lines(i + 1), texts[i])