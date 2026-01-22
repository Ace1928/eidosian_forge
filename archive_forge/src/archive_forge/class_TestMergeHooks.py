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
class TestMergeHooks(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.tree_a = self.make_branch_and_tree('tree_a')
        self.build_tree_contents([('tree_a/file', b'content_1')])
        self.tree_a.add('file', ids=b'file-id')
        self.tree_a.commit('added file')
        self.tree_b = self.tree_a.controldir.sprout('tree_b').open_workingtree()
        self.build_tree_contents([('tree_b/file', b'content_2')])
        self.tree_b.commit('modify file')

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

    def test_post_merge_hook_called(self):
        calls = []

        def factory(merger):
            self.assertIsInstance(merger, _mod_merge.Merge3Merger)
            calls.append(merger)
        _mod_merge.Merger.hooks.install_named_hook('post_merge', factory, 'test factory')
        self.tree_a.merge_from_branch(self.tree_b.branch)
        self.assertFileEqual(b'content_2', 'tree_a/file')
        self.assertLength(1, calls)