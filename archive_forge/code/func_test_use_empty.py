import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def test_use_empty(self):
    ignores._set_user_ignores([])
    ignore_path = bedding.user_ignore_config_path()
    self.check_file_contents(ignore_path, b'')
    self.assertEqual(set(), ignores.get_user_ignores())