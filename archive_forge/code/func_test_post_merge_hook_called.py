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
def test_post_merge_hook_called(self):
    calls = []

    def factory(merger):
        self.assertIsInstance(merger, _mod_merge.Merge3Merger)
        calls.append(merger)
    _mod_merge.Merger.hooks.install_named_hook('post_merge', factory, 'test factory')
    self.tree_a.merge_from_branch(self.tree_b.branch)
    self.assertFileEqual(b'content_2', 'tree_a/file')
    self.assertLength(1, calls)