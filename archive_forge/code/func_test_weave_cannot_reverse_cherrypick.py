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
def test_weave_cannot_reverse_cherrypick(self):
    this_tree, other_tree = self.prepare_cherrypick()
    merger = _mod_merge.Merger.from_revision_ids(this_tree, b'rev2b', b'rev3b', other_tree.branch)
    merger.merge_type = _mod_merge.WeaveMerger
    self.assertRaises(errors.CannotReverseCherrypick, merger.do_merge)