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
def test_sprout_recursive(self):
    tree = self.make_branch_and_tree('tree1')
    sub_tree = self.make_branch_and_tree('tree1/subtree')
    sub_tree.set_root_id(b'subtree-root')
    tree.add_reference(sub_tree)
    tree.set_reference_info('subtree', sub_tree.branch.user_url)
    self.build_tree(['tree1/subtree/file'])
    sub_tree.add('file')
    tree.commit('Initial commit')
    tree2 = tree.controldir.sprout('tree2').open_workingtree()
    tree2.lock_read()
    self.addCleanup(tree2.unlock)
    self.assertPathExists('tree2/subtree/file')
    self.assertEqual('tree-reference', tree2.kind('subtree'))