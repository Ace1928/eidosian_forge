from breezy import conflicts, tests
from breezy.bzr import conflicts as _mod_bzr_conflicts
from breezy.tests import KnownFailure, script
from breezy.tests.blackbox import test_conflicts
def test_resolve_one_by_one(self):
    self.run_script('$ cd branch\n$ brz conflicts\nText conflict in my_other_file\nPath conflict: mydir3 / mydir2\nText conflict in myfile\n$ brz resolve myfile\n2>1 conflict resolved, 2 remaining\n$ brz resolve my_other_file\n2>1 conflict resolved, 1 remaining\n$ brz resolve mydir2\n2>1 conflict resolved, 0 remaining\n')