import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def test_parse_empty(self):
    ignored = ignores.parse_ignore_file(BytesIO(b''))
    self.assertEqual(set(), ignored)