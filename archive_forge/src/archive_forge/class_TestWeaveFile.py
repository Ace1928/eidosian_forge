from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class TestWeaveFile(TestCaseInTempDir):

    def test_empty_file(self):
        with open('empty.weave', 'wb+') as f:
            self.assertRaises(WeaveFormatError, read_weave, f)