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
def test_same_lca_resolution_one_side_updates_content(self):
    builder = self.get_builder()
    builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'A content\n'))], revision_id=b'A-id')
    builder.build_snapshot([b'A-id'], [('modify', ('foo', b'B content\n'))], revision_id=b'B-id')
    builder.build_snapshot([b'A-id'], [('modify', ('foo', b'C content\n'))], revision_id=b'C-id')
    builder.build_snapshot([b'C-id', b'B-id'], [], revision_id=b'E-id')
    builder.build_snapshot([b'B-id', b'C-id'], [('modify', ('foo', b'C content\n'))], revision_id=b'D-id')
    builder.build_snapshot([b'D-id'], [('modify', ('foo', b'F content\n'))], revision_id=b'F-id')
    merge_obj = self.make_merge_obj(builder, b'E-id')
    entries = list(merge_obj._entries_lca())
    self.expectFailure("We don't detect that LCA resolution was the same on both sides", self.assertEqual, [], entries)