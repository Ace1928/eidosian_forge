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
def test_plan_merge(self):
    self.setup_plan_merge()
    plan = self.plan_merge_vf.plan_merge(b'B', b'C')
    self.assertEqual([('new-b', b'f\n'), ('unchanged', b'a\n'), ('killed-a', b'b\n'), ('killed-b', b'c\n'), ('new-a', b'e\n'), ('new-a', b'h\n'), ('new-a', b'g\n'), ('new-b', b'g\n')], list(plan))