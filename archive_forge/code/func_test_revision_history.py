from breezy import branch, tests
def test_revision_history(self):
    """No location"""
    self._build_branch()
    self._check_revision_history(working_dir='test')