import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def test_parse_non_utf8(self):
    """Lines with non utf 8 characters should be discarded."""
    ignored = ignores.parse_ignore_file(BytesIO(b'utf8filename_a\ninvalid utf8\x80\nutf8filename_b\n'))
    self.assertEqual({'utf8filename_a', 'utf8filename_b'}, ignored)