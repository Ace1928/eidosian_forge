import os
import sys
from ... import (branch, controldir, errors, repository, upgrade, urlutils,
from ...bzr import bzrdir
from ...bzr.tests import test_bundle
from ...osutils import getcwd
from ...tests import TestCaseWithTransport
from ...tests.test_sftp_transport import TestCaseWithSFTPServer
from .branch import BzrBranchFormat4
from .bzrdir import BzrDirFormat5, BzrDirFormat6
def test_upgrade_with_ghosts(self):
    """Upgrade v0.0.4 tree containing ghost references.

        That is, some of the parents of revisions mentioned in the branch
        aren't present in the branch's storage.

        This shouldn't normally happen in branches created entirely in
        bzr, but can happen in branches imported from baz and arch, or from
        other systems, where the importer knows about a revision but not
        its contents."""
    eq = self.assertEqual
    self.build_tree_contents(_ghost_template)
    upgrade.upgrade('.')
    b = branch.Branch.open('.')
    self.addCleanup(b.lock_read().unlock)
    revision_id = b._revision_history()[1]
    rev = b.repository.get_revision(revision_id)
    eq(len(rev.parent_ids), 2)
    eq(rev.parent_ids[1], b'wibble@wobble-2')