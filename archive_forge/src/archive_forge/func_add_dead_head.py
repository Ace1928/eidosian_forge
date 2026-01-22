from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def add_dead_head(self, tree):
    revno, revision_id = tree.branch.last_revision_info()
    tree.commit('Dead head', rev_id=b'dead-head-id')
    tree.branch.set_last_revision_info(revno, revision_id)
    tree.set_last_revision(revision_id)