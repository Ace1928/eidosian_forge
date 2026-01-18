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
def test_two_roots(self):
    """Merge base is sane when two unrelated branches are merged"""
    wt1, br2 = self.test_pending_with_null()
    wt1.commit('blah')
    wt1.lock_read()
    try:
        last = wt1.branch.last_revision()
        last2 = br2.last_revision()
        graph = wt1.branch.repository.get_graph()
        self.assertEqual(last2, graph.find_unique_lca(last, last2))
    finally:
        wt1.unlock()