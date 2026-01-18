import os
import subprocess
import sys
import breezy.branch
import breezy.bzr.branch
from ... import (branch, bzr, config, controldir, errors, help_topics, lock,
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ... import urlutils, win32utils
from ...errors import (NotBranchError, UnknownFormatError,
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from ...transport import memory, pathfilter
from ...transport.http.urllib import HttpTransport
from ...transport.nosmart import NoSmartTransportDecorator
from ...transport.readonly import ReadonlyTransportDecorator
from .. import branch as bzrbranch
from .. import (bzrdir, knitpack_repo, knitrepo, remote, workingtree_3,
from ..fullhistory import BzrBranchFormat5
def test_open_tree_or_branch(self):
    self.make_branch_and_tree('topdir')
    tree, branch = bzrdir.BzrDir.open_tree_or_branch('topdir')
    self.assertEqual(os.path.realpath('topdir'), os.path.realpath(tree.basedir))
    self.assertEqual(os.path.realpath('topdir'), self.local_branch_path(branch))
    self.assertIs(tree.controldir, branch.controldir)
    tree, branch = bzrdir.BzrDir.open_tree_or_branch(self.get_readonly_url('topdir'))
    self.assertEqual(None, tree)
    self.make_branch('topdir/foo')
    tree, branch = bzrdir.BzrDir.open_tree_or_branch('topdir/foo')
    self.assertIs(tree, None)
    self.assertEqual(os.path.realpath('topdir/foo'), self.local_branch_path(branch))