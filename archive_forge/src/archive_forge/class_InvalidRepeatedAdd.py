from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class InvalidRepeatedAdd(TestBase):

    def runTest(self):
        k = Weave()
        k.add_lines(b'basis', [], TEXT_0)
        k.add_lines(b'text0', [], TEXT_0)
        self.assertRaises(errors.RevisionAlreadyPresent, k.add_lines, b'text0', [], [b'not the same text'])
        self.assertRaises(errors.RevisionAlreadyPresent, k.add_lines, b'text0', [b'basis'], TEXT_0)