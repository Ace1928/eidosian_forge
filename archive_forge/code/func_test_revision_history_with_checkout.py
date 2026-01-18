from breezy import branch, tests
def test_revision_history_with_checkout(self):
    """With a repository branch checkout location."""
    self._build_branch()
    self.run_bzr('init-shared-repo repo')
    self.run_bzr('branch test repo/test')
    self.run_bzr('checkout repo/test test-checkout')
    self._check_revision_history('test-checkout')