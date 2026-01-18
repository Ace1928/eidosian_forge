import os
from io import BytesIO
from .. import (conflicts, errors, symbol_versioning, trace, transport,
from ..bzr import bzrdir
from ..bzr import conflicts as _mod_bzr_conflicts
from ..bzr import workingtree as bzrworkingtree
from ..bzr import workingtree_3, workingtree_4
from ..lock import write_locked
from ..lockdir import LockDir
from ..tree import TreeDirectory, TreeEntry, TreeFile, TreeLink
from . import TestCase, TestCaseWithTransport, TestSkipped
from .features import SymlinkFeature
def test_revert_conflicts_recursive(self):
    this_tree = self.make_branch_and_tree('this-tree')
    self.build_tree_contents([('this-tree/foo/',), ('this-tree/foo/bar', b'bar')])
    this_tree.add(['foo', 'foo/bar'])
    this_tree.commit('created foo/bar')
    other_tree = this_tree.controldir.sprout('other-tree').open_workingtree()
    self.build_tree_contents([('other-tree/foo/bar', b'baz')])
    other_tree.commit('changed bar')
    self.build_tree_contents([('this-tree/foo/bar', b'qux')])
    this_tree.commit('changed qux')
    this_tree.merge_from_branch(other_tree.branch)
    self.assertEqual(1, len(this_tree.conflicts()))
    this_tree.revert(['foo'])
    self.assertEqual(0, len(this_tree.conflicts()))