import sys
from unittest import TestLoader, TestSuite
from breezy.tests import TestCaseWithTransport
def test_repo_has_key(self):
    self.make_branch_and_tree('repo')
    out, err = self.run_bzr('repo-has-key repo revisions revid', retcode=1)
    self.assertEqual(out, 'False\n')
    self.assertEqual(err, '')