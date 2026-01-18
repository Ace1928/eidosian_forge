from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_reference_info_caching_write_locked(self):
    gets = []
    branch = self.make_branch('branch')
    branch.lock_write()
    self.instrument_branch(branch, gets)
    self.addCleanup(branch.unlock)
    branch._set_all_reference_info({'path2': ('location2', b'file-id')})
    location, file_id = branch.get_reference_info('path2')
    self.assertEqual(0, len(gets))
    self.assertEqual(b'file-id', file_id)
    self.assertEqual('location2', location)