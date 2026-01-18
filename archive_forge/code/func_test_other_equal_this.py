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
def test_other_equal_this(self):
    """Test when other and this are identical."""
    self.assertLCAMultiWay('this', 'bval', ['bval', 'bval'], 'oval', 'oval')
    self.assertLCAMultiWay('this', 'bval', ['lcaval', 'lcaval'], 'oval', 'oval')
    self.assertLCAMultiWay('this', 'bval', ['cval', 'dval'], 'oval', 'oval')
    self.assertLCAMultiWay('this', 'bval', [None, 'lcaval'], 'oval', 'oval')
    self.assertLCAMultiWay('this', None, [None, 'lcaval'], 'oval', 'oval')
    self.assertLCAMultiWay('this', None, ['lcaval', 'lcaval'], 'oval', 'oval')
    self.assertLCAMultiWay('this', None, ['cval', 'dval'], 'oval', 'oval')
    self.assertLCAMultiWay('this', None, ['cval', 'dval'], None, None)
    self.assertLCAMultiWay('this', None, ['cval', 'dval', 'eval', 'fval'], 'oval', 'oval')