import codecs
import sys
from io import BytesIO, StringIO
from os import chdir, mkdir, rmdir, unlink
import breezy.branch
from breezy.bzr import bzrdir, conflicts
from ... import errors, osutils, status
from ...osutils import pathjoin
from ...revisionspec import RevisionSpec
from ...status import show_tree_status
from ...workingtree import WorkingTree
from .. import TestCaseWithTransport, TestSkipped
def test_status_write_lock(self):
    """Test that status works without fetching history and
        having a write lock.

        See https://bugs.launchpad.net/bzr/+bug/149270
        """
    mkdir('branch1')
    wt = self.make_branch_and_tree('branch1')
    b = wt.branch
    wt.commit('Empty commit 1')
    wt2 = b.controldir.sprout('branch2').open_workingtree()
    wt2.commit('Empty commit 2')
    out, err = self.run_bzr('status branch1 -rbranch:branch2')
    self.assertEqual('', out)