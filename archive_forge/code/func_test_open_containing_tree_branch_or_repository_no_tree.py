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
def test_open_containing_tree_branch_or_repository_no_tree(self):
    self.make_branch('branch')
    tree, branch, repo, relpath = bzrdir.BzrDir.open_containing_tree_branch_or_repository('branch/foo')
    self.assertEqual(tree, None)
    self.assertEqual(os.path.realpath('branch'), self.local_branch_path(branch))
    self.assertEqual(osutils.realpath(os.path.join('branch', '.bzr', 'repository')), repo.controldir.transport.local_abspath('repository'))
    self.assertEqual(relpath, 'foo')