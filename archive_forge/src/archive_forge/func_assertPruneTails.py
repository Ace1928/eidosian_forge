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
def assertPruneTails(self, pruned_map, tails, parent_map):
    child_map = {}
    for key, parent_keys in parent_map.items():
        child_map.setdefault(key, [])
        for pkey in parent_keys:
            child_map.setdefault(pkey, []).append(key)
    _PlanMerge._prune_tails(parent_map, child_map, tails)
    self.assertEqual(pruned_map, parent_map)