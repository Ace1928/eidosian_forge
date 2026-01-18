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
def test_this_in_lca(self):
    self.assertLCAMultiWay('other', 'bval', ['lca1val', 'lca2val'], 'oval', 'lca1val')
    self.assertLCAMultiWay('other', 'bval', ['lca1val', 'lca2val'], 'oval', 'lca2val')
    self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val'], 'oval', 'lca1val', allow_overriding_lca=False)
    self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val'], 'oval', 'lca2val', allow_overriding_lca=False)
    self.assertLCAMultiWay('other', 'bval', ['lca1val', 'lca2val', 'lca3val'], 'bval', 'lca3val')
    self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val', 'lca3val'], 'bval', 'lca3val', allow_overriding_lca=False)