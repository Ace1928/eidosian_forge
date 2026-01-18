import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
def test_root_entry_has_revision(self):
    tree = self.make_branch_and_tree('.')
    revid = tree.commit('message')
    rev_tree = tree.branch.repository.revision_tree(tree.last_revision())
    rev_tree.lock_read()
    self.addCleanup(rev_tree.unlock)
    self.assertEqual(revid, rev_tree.get_file_revision(''))