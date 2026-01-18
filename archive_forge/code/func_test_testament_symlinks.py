import os
from breezy import osutils
from breezy.bzr.testament import StrictTestament, StrictTestament3, Testament
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import SymlinkFeature
def test_testament_symlinks(self):
    """Testament containing symlink (where possible)"""
    self.requireFeature(SymlinkFeature(self.test_dir))
    os.symlink('wibble/linktarget', 'link')
    self.wt.add(['link'], ids=[b'link-id'])
    self.wt.commit(message='add symlink', timestamp=1129025493, timezone=36000, rev_id=b'test@user-3', committer='test@user')
    t = self.from_revision(self.b.repository, b'test@user-3')
    self.assertEqualDiff(t.as_text(), self.expected('rev_3'))