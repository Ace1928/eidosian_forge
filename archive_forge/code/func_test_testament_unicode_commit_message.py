import os
from breezy import osutils
from breezy.bzr.testament import StrictTestament, StrictTestament3, Testament
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import SymlinkFeature
def test_testament_unicode_commit_message(self):
    self.wt.commit(message='non-ascii commit © me', timestamp=1129025493, timezone=36000, rev_id=b'test@user-3', committer='Erik Bågfors <test@user>', revprops={'uni': 'µ'})
    t = self.from_revision(self.b.repository, b'test@user-3')
    self.assertEqualDiff(self.expected('sample_unicode').encode('utf-8'), t.as_text())