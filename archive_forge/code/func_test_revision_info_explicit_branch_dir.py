import os
from breezy.errors import CommandError, NoSuchRevision
from breezy.tests import TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_revision_info_explicit_branch_dir(self):
    """Test that 'brz revision-info' honors the '-d' option."""
    wt = self.make_branch_and_tree('branch')
    wt.commit('Commit one', rev_id=b'a@r-0-1')
    self.check_output('1 a@r-0-1\n', 'revision-info -d branch')