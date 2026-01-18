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
def test_renamed_symlink(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    wt = self.make_branch_and_tree('path')
    wt.lock_write()
    self.addCleanup(wt.unlock)
    os.symlink('bar', 'path/foo')
    wt.add(['foo'], ids=[b'foo-id'])
    wt.commit('A add symlink', rev_id=b'A-id')
    wt.rename_one('foo', 'barry')
    wt.commit('B foo => barry', rev_id=b'B-id')
    wt.set_last_revision(b'A-id')
    wt.branch.set_last_revision_info(1, b'A-id')
    wt.revert()
    wt.commit('C', rev_id=b'C-id')
    wt.merge_from_branch(wt.branch, b'B-id')
    self.assertEqual('barry', wt.id2path(b'foo-id'))
    self.assertEqual('bar', wt.get_symlink_target('barry'))
    wt.commit('E merges C & B', rev_id=b'E-id')
    wt.rename_one('barry', 'blah')
    wt.commit('F barry => blah', rev_id=b'F-id')
    wt.set_last_revision(b'B-id')
    wt.branch.set_last_revision_info(2, b'B-id')
    wt.revert()
    wt.merge_from_branch(wt.branch, b'C-id')
    wt.commit('D merges B & C', rev_id=b'D-id')
    self.assertEqual('barry', wt.id2path(b'foo-id'))
    merger = _mod_merge.Merger.from_revision_ids(wt, b'F-id')
    merger.merge_type = _mod_merge.Merge3Merger
    merge_obj = merger.make_merger()
    root_id = wt.path2id('')
    entries = list(merge_obj._entries_lca())
    self.assertEqual([(b'foo-id', False, (('foo', ['barry', 'foo']), 'blah', 'barry'), ((root_id, [root_id, root_id]), root_id, root_id), (('foo', ['barry', 'foo']), 'blah', 'barry'), ((False, [False, False]), False, False), False)], entries)
    conflicts = wt.merge_from_branch(wt.branch, to_revision=b'F-id')
    self.assertEqual(0, len(conflicts))
    self.assertEqual('blah', wt.id2path(b'foo-id'))