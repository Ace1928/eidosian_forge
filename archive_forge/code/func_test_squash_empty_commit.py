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
def test_squash_empty_commit(self):
    params = {b'include_paths': None, b'exclude_paths': None}
    self.assertFiltering(_SAMPLE_EMPTY_COMMIT, params, b'blob\nmark :1\ndata 4\nfoo\ncommit refs/heads/master\nmark :2\ncommitter Joe <joe@example.com> 1234567890 +1000\ndata 14\nInitial import\nM 644 :1 COPYING\n')