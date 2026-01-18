import os
from breezy import osutils
from breezy.bzr.testament import StrictTestament, StrictTestament3, Testament
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import SymlinkFeature
def test_testament_revprops(self):
    """Testament to revision with extra properties"""
    props = {'flavor': 'sour cherry\ncream cheese', 'size': 'medium', 'empty': ''}
    self.wt.commit(message='revision with properties', timestamp=1129025493, timezone=36000, rev_id=b'test@user-3', committer='test@user', revprops=props)
    t = self.from_revision(self.b.repository, b'test@user-3')
    self.assertEqualDiff(t.as_text(), self.expected('rev_props'))