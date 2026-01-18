from breezy import conflicts, tests
from breezy.bzr import conflicts as _mod_bzr_conflicts
from breezy.tests import KnownFailure, script
from breezy.tests.blackbox import test_conflicts
def test_resolve_all(self):
    self.run_script('$ cd branch\n$ brz resolve --all\n2>3 conflicts resolved, 0 remaining\n$ brz conflicts\n')