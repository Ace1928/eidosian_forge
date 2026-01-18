from breezy import branch, config, errors, tests
from ..test_bedding import override_whoami
def test_whoami_nonbranch_directory(self):
    """Test --directory mentioning a non-branch directory."""
    wt = self.build_tree(['subdir/'])
    out, err = self.run_bzr('whoami --directory subdir', retcode=3)
    self.assertContainsRe(err, 'ERROR: Not a branch')