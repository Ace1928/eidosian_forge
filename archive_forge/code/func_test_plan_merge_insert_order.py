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
def test_plan_merge_insert_order(self):
    """Weave merges are sensitive to the order of insertion.

        Specifically for overlapping regions, it effects which region gets put
        'first'. And when a user resolves an overlapping merge, if they use the
        same ordering, then the lines match the parents, if they don't only
        *some* of the lines match.
        """
    self.add_rev(b'root', b'A', [], b'abcdef')
    self.add_rev(b'root', b'B', [b'A'], b'abwxcdef')
    self.add_rev(b'root', b'C', [b'A'], b'abyzcdef')
    self.add_rev(b'root', b'D', [b'B', b'C'], b'abwxyzcdef')
    self.add_rev(b'root', b'E', [b'C', b'B'], b'abnocdef')
    self.add_rev(b'root', b'F', [b'C'], b'abpqcdef')
    plan = self.plan_merge_vf.plan_merge(b'D', b'E')
    self.assertEqual([('unchanged', b'a\n'), ('unchanged', b'b\n'), ('killed-b', b'w\n'), ('killed-b', b'x\n'), ('killed-b', b'y\n'), ('killed-b', b'z\n'), ('new-b', b'n\n'), ('new-b', b'o\n'), ('unchanged', b'c\n'), ('unchanged', b'd\n'), ('unchanged', b'e\n'), ('unchanged', b'f\n')], list(plan))
    plan = self.plan_merge_vf.plan_merge(b'E', b'D')
    self.assertEqual([('unchanged', b'a\n'), ('unchanged', b'b\n'), ('new-b', b'w\n'), ('new-b', b'x\n'), ('killed-a', b'y\n'), ('killed-a', b'z\n'), ('killed-both', b'w\n'), ('killed-both', b'x\n'), ('new-a', b'n\n'), ('new-a', b'o\n'), ('unchanged', b'c\n'), ('unchanged', b'd\n'), ('unchanged', b'e\n'), ('unchanged', b'f\n')], list(plan))