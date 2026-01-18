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
def test_plan_lca_merge_with_null(self):
    self.add_version((b'root', b'A'), [], b'ab')
    self.add_version((b'root', b'B'), [], b'bc')
    plan = self.plan_merge_vf.plan_lca_merge(b'A', b'B')
    self.assertEqual([('new-a', b'a\n'), ('unchanged', b'b\n'), ('new-b', b'c\n')], list(plan))