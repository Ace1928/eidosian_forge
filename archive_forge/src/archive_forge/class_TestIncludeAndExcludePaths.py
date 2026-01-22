from io import BytesIO
from unittest import TestCase
from fastimport import (
from fastimport.processors import (
from :2
from :2
from :100
from :101
from :100
from :100
from :100
from :100
from :101
from :100
from :100
from :102
from :102
from :102
from :100
from :102
from :100
from :102
from :100
from :102
from :102
from :102
from :100
from :102
from :100
from :100
from :100
from :100
from :100
from :102
from :101
from :102
from :101
import
from :999
from :3
import
from :999
from :3
import
from :999
from :3
import
from :999
from :3
import
from :999
from :3
class TestIncludeAndExcludePaths(TestCaseWithFiltering):

    def test_included_dir_and_excluded_file(self):
        params = {b'include_paths': [b'doc/'], b'exclude_paths': [b'doc/index.txt']}
        self.assertFiltering(_SAMPLE_WITH_DIR, params, b'blob\nmark :1\ndata 9\nWelcome!\ncommit refs/heads/master\nmark :100\ncommitter a <b@c> 1234798653 +0000\ndata 4\ntest\nM 644 :1 README.txt\nblob\nmark :3\ndata 19\nWelcome!\nmy friend\ncommit refs/heads/master\nmark :102\ncommitter d <b@c> 1234798653 +0000\ndata 8\ntest\ning\nfrom :100\nM 644 :3 README.txt\n')