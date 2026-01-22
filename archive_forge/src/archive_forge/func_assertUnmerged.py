from .. import missing, tests
from ..missing import iter_log_revisions
from . import TestCaseWithTransport
def assertUnmerged(self, local, remote, local_branch, remote_branch, restrict='all', include_merged=False, backward=False, local_revid_range=None, remote_revid_range=None):
    """Check the output of find_unmerged_mainline_revisions"""
    local_extra, remote_extra = missing.find_unmerged(local_branch, remote_branch, restrict, include_merged=include_merged, backward=backward, local_revid_range=local_revid_range, remote_revid_range=remote_revid_range)
    self.assertEqual(local, local_extra)
    self.assertEqual(remote, remote_extra)