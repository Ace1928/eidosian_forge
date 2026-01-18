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
def test_other_reverted_content_to_base(self):
    builder = self.get_builder()
    builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'base content\n'))], revision_id=b'A-id')
    builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
    builder.build_snapshot([b'A-id'], [('modify', ('foo', b'B content\n'))], revision_id=b'B-id')
    builder.build_snapshot([b'C-id', b'B-id'], [('modify', ('foo', b'B content\n'))], revision_id=b'E-id')
    builder.build_snapshot([b'E-id'], [('modify', ('foo', b'base content\n'))], revision_id=b'F-id')
    builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
    wt, conflicts = self.do_merge(builder, b'F-id')
    self.assertEqual([], conflicts)
    self.assertEqual(b'base content\n', wt.get_file_text('foo'))