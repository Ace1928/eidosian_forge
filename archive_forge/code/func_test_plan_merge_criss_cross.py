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
def test_plan_merge_criss_cross(self):
    self.add_rev(b'root', b'XX', [], b'qrs')
    self.add_rev(b'root', b'A', [b'XX'], b'abcdef')
    self.add_rev(b'root', b'B', [b'A'], b'axcdef')
    self.add_rev(b'root', b'C', [b'B'], b'axcdefg')
    self.add_rev(b'root', b'D', [b'B'], b'haxcdef')
    self.add_rev(b'root', b'E', [b'A'], b'abcdyf')
    self.add_rev(b'root', b'F', [b'C', b'D', b'E'], b'haxcdyfg')
    self.add_rev(b'root', b'G', [b'C', b'D', b'E'], b'hazcdyfg')
    plan = self.plan_merge_vf.plan_merge(b'F', b'G')
    self.assertEqual([('unchanged', b'h\n'), ('unchanged', b'a\n'), ('killed-base', b'b\n'), ('killed-b', b'x\n'), ('new-b', b'z\n'), ('unchanged', b'c\n'), ('unchanged', b'd\n'), ('killed-base', b'e\n'), ('unchanged', b'y\n'), ('unchanged', b'f\n'), ('unchanged', b'g\n')], list(plan))
    plan = self.plan_merge_vf.plan_lca_merge(b'F', b'G')
    self.assertEqual([('unchanged', b'h\n'), ('unchanged', b'a\n'), ('conflicted-a', b'x\n'), ('new-b', b'z\n'), ('unchanged', b'c\n'), ('unchanged', b'd\n'), ('unchanged', b'y\n'), ('unchanged', b'f\n'), ('unchanged', b'g\n')], list(plan))