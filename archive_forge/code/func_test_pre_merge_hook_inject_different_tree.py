import contextlib
import os
from .. import branch as _mod_branch
from .. import conflicts, errors, memorytree
from .. import merge as _mod_merge
from .. import option
from .. import revision as _mod_revision
from .. import tests, transform
from ..bzr import inventory, knit, versionedfile
from ..bzr.conflicts import (ContentsConflict, DeletingParent, MissingParent,
from ..conflicts import ConflictList
from ..errors import NoCommits, UnrelatedBranches
from ..merge import _PlanMerge, merge_inner, transform_tree
from ..osutils import basename, file_kind, pathjoin
from ..workingtree import PointlessMerge, WorkingTree
from . import (TestCaseWithMemoryTransport, TestCaseWithTransport, features,
def test_pre_merge_hook_inject_different_tree(self):
    tree_c = self.tree_b.controldir.sprout('tree_c').open_workingtree()
    self.build_tree_contents([('tree_c/file', b'content_3')])
    tree_c.commit('more content')
    calls = []

    def factory(merger):
        self.assertIsInstance(merger, _mod_merge.Merge3Merger)
        merger.other_tree = tree_c
        calls.append(merger)
    _mod_merge.Merger.hooks.install_named_hook('pre_merge', factory, 'test factory')
    self.tree_a.merge_from_branch(self.tree_b.branch)
    self.assertFileEqual(b'content_3', 'tree_a/file')
    self.assertLength(1, calls)