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
def test_rmdir_conflict(self):
    tree_a = self.make_branch_and_tree('a')
    self.build_tree(['a/b/'])
    tree_a.add('b', ids=b'b-id')
    tree_a.commit('added b')
    base_tree = tree_a.branch.repository.revision_tree(tree_a.last_revision())
    tree_z = tree_a.controldir.sprout('z').open_workingtree()
    self.build_tree(['a/b/c'])
    tree_a.add('b/c')
    tree_a.commit('added c')
    os.rmdir('z/b')
    tree_z.commit('removed b')
    merge_inner(tree_z.branch, tree_a, base_tree, this_tree=tree_z)
    self.assertEqual([MissingParent('Created directory', 'b', b'b-id'), UnversionedParent('Versioned directory', 'b', b'b-id')], tree_z.conflicts())
    merge_inner(tree_a.branch, tree_z.basis_tree(), base_tree, this_tree=tree_a)
    self.assertEqual([DeletingParent('Not deleting', 'b', b'b-id'), UnversionedParent('Versioned directory', 'b', b'b-id')], tree_a.conflicts())