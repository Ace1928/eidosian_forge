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
def test_plan_merge_no_common_ancestor(self):
    self.add_rev(b'root', b'A', [], b'abc')
    self.add_rev(b'root', b'B', [], b'xyz')
    my_plan = _PlanMerge(b'A', b'B', self.plan_merge_vf, (b'root',))
    self.assertEqual([('new-a', b'a\n'), ('new-a', b'b\n'), ('new-a', b'c\n'), ('new-b', b'x\n'), ('new-b', b'y\n'), ('new-b', b'z\n')], list(my_plan.plan_merge()))