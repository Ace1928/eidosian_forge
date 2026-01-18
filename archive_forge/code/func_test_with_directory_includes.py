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
def test_with_directory_includes(self):
    params = {b'include_paths': [b'data/'], b'exclude_paths': None, b'squash_empty_commits': False}
    self.assertFiltering(_SAMPLE_FROM_MERGE_COMMIT, params, b'commit refs/heads/master\nmark :3\ncommitter Joe <joe@example.com> 1234567890 +1000\ndata 6\nimport\nblob\nmark :2\ndata 4\nbar\ncommit refs/heads/master\nmark :4\ncommitter Joe <joe@example.com> 1234567890 +1000\ndata 19\nunknown from commit\nfrom :999\nM 644 :2 DATA\nblob\nmark :99\ndata 4\nbar\ncommit refs/heads/master\nmark :5\ncommitter Joe <joe@example.com> 1234567890 +1000\ndata 12\nmerge commit\nfrom :3\nmerge :4\nmerge :1001\nM 644 :99 DATA2\n')