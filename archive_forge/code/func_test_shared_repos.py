from breezy.controldir import ControlDir
from breezy.tests import TestCaseWithTransport
def test_shared_repos(self):
    self.make_repository('a', shared=True)
    ControlDir.create_branch_convenience('a/branch1')
    b = ControlDir.create_branch_convenience('a/branch2')
    b.create_checkout(lightweight=True, to_location='b')
    out, err = self.run_bzr('branches b')
    self.assertEqual(out, '  branch1\n* branch2\n')