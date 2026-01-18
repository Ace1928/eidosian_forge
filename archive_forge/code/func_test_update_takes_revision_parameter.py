import errno
import os
from io import StringIO
from ... import branch as _mod_branch
from ... import config, controldir, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import bzrdir
from ...bzr.conflicts import ConflictList, ContentsConflict, TextConflict
from ...bzr.inventory import Inventory
from ...bzr.workingtree import InventoryWorkingTree
from ...errors import PathsNotVersionedError, UnsupportedOperation
from ...mutabletree import MutableTree
from ...osutils import getcwd, pathjoin, supports_symlinks
from ...tree import TreeDirectory, TreeFile, TreeLink
from ...workingtree import SettingFileIdUnsupported, WorkingTree
from .. import TestNotApplicable, TestSkipped, features
from . import TestCaseWithWorkingTree
def test_update_takes_revision_parameter(self):
    wt = self.make_branch_and_tree('wt')
    self.build_tree_contents([('wt/a', b'old content')])
    wt.add(['a'])
    rev1 = wt.commit('first master commit')
    self.build_tree_contents([('wt/a', b'new content')])
    rev2 = wt.commit('second master commit')
    conflicts = wt.update(revision=rev1)
    self.assertFileEqual(b'old content', 'wt/a')
    self.assertEqual([rev1], wt.get_parent_ids())