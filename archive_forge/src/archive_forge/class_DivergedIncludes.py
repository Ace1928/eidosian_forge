from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class DivergedIncludes(TestBase):
    """Weave with two diverged texts based on version 0.
    """

    def runTest(self):
        k = Weave()
        k._names = [b'0', b'1', b'2']
        k._name_map = {b'0': 0, b'1': 1, b'2': 2}
        k._parents = [frozenset(), frozenset([0]), frozenset([0])]
        k._weave = [(b'{', 0), b'first line', (b'}', 0), (b'{', 1), b'second line', (b'}', 1), (b'{', 2), b'alternative second line', (b'}', 2)]
        k._sha1s = [sha_string(b'first line'), sha_string(b'first linesecond line'), sha_string(b'first linealternative second line')]
        self.assertEqual(k.get_lines(0), [b'first line'])
        self.assertEqual(k.get_lines(1), [b'first line', b'second line'])
        self.assertEqual(k.get_lines(b'2'), [b'first line', b'alternative second line'])
        self.assertEqual(set(k.get_ancestry([b'2'])), {b'0', b'2'})