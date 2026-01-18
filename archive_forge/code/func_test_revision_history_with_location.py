from breezy import branch, tests
def test_revision_history_with_location(self):
    """With a specified location."""
    self._build_branch()
    self._check_revision_history('test')