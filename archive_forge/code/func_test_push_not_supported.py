from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
def test_push_not_supported(self):
    source_tree = self.make_branch_and_tree('source')
    target_tree = self.make_branch_and_tree('target', format=DummyForeignVcsDirFormat())
    self.assertRaises(errors.NoRoundtrippingSupport, source_tree.branch.push, target_tree.branch)