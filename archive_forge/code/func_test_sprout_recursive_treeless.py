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
def test_sprout_recursive_treeless(self):
    tree = self.make_branch_and_tree('tree1', format='development-subtree')
    sub_tree = self.make_branch_and_tree('tree1/subtree', format='development-subtree')
    tree.add_reference(sub_tree)
    tree.set_reference_info('subtree', sub_tree.branch.user_url)
    self.build_tree(['tree1/subtree/file'])
    sub_tree.add('file')
    tree.commit('Initial commit')
    tree.branch.get_config_stack().set('transform.orphan_policy', 'move')
    tree.controldir.destroy_workingtree()
    repo = self.make_repository('repo', shared=True, format='development-subtree')
    repo.set_make_working_trees(False)
    tree.controldir.sprout('repo/tree2')
    self.assertPathExists('repo/tree2/subtree')
    self.assertPathDoesNotExist('repo/tree2/subtree/file')