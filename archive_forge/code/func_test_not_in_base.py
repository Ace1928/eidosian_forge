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
def test_not_in_base(self):
    builder = self.get_builder()
    builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None))], revision_id=b'A-id')
    builder.build_snapshot([b'A-id'], [('add', ('foo', b'foo-id', 'file', b'a\nb\nc\n'))], revision_id=b'B-id')
    builder.build_snapshot([b'A-id'], [('add', ('bar', b'bar-id', 'file', b'd\ne\nf\n'))], revision_id=b'C-id')
    builder.build_snapshot([b'B-id', b'C-id'], [('add', ('bar', b'bar-id', 'file', b'd\ne\nf\n'))], revision_id=b'D-id')
    builder.build_snapshot([b'C-id', b'B-id'], [('add', ('foo', b'foo-id', 'file', b'a\nb\nc\n'))], revision_id=b'E-id')
    builder.build_snapshot([b'E-id', b'D-id'], [('modify', ('bar', b'd\ne\nf\nG\n'))], revision_id=b'G-id')
    builder.build_snapshot([b'D-id', b'E-id'], [], revision_id=b'F-id')
    merge_obj = self.make_merge_obj(builder, b'G-id')
    self.assertEqual([b'D-id', b'E-id'], [t.get_revision_id() for t in merge_obj._lca_trees])
    self.assertEqual(b'A-id', merge_obj.base_tree.get_revision_id())
    entries = list(merge_obj._entries_lca())
    root_id = b'a-root-id'
    self.assertEqual([(b'bar-id', True, ((None, ['bar', 'bar']), 'bar', 'bar'), ((None, [root_id, root_id]), root_id, root_id), ((None, ['bar', 'bar']), 'bar', 'bar'), ((None, [False, False]), False, False), False)], entries)