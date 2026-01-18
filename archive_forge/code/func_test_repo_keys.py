import sys
from unittest import TestLoader, TestSuite
from breezy.tests import TestCaseWithTransport
def test_repo_keys(self):
    self.make_branch_and_tree('a')
    out, err = self.run_bzr('repo-keys a texts')
    self.assertEqual(out, '')
    self.assertEqual(err, '')