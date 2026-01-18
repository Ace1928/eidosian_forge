import os
from breezy.tests.features import SymlinkFeature
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_walkdir_from_missing_dir(self):
    """Doing a walkdir when the requested prefix is missing but on disk."""
    self._test_walkdir(self.missing, 'missing dir')