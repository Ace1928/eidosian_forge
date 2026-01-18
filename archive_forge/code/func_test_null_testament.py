import os
from breezy import osutils
from breezy.bzr.testament import StrictTestament, StrictTestament3, Testament
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import SymlinkFeature
def test_null_testament(self):
    """Testament for a revision with no contents."""
    t = self.from_revision(self.b.repository, b'test@user-1')
    ass = self.assertTrue
    eq = self.assertEqual
    ass(isinstance(t, Testament))
    eq(t.revision_id, b'test@user-1')
    eq(t.committer, 'test@user')
    eq(t.timestamp, 1129025423)
    eq(t.timezone, 0)