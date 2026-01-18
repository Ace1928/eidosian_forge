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
def test_tag_retention(self):
    params = {b'include_paths': [b'NEWS']}
    self.assertFiltering(_SAMPLE_WITH_TAGS, params, b'blob\nmark :2\ndata 17\nLife\nis\ngood ...\ncommit refs/heads/master\nmark :101\ncommitter a <b@c> 1234798653 +0000\ndata 8\ntest\ning\nM 644 :2 NEWS\ntag v0.2\nfrom :101\ntagger d <b@c> 1234798653 +0000\ndata 12\nrelease v0.2\n')