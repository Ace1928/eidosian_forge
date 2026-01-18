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
def test_symlink_this_changed_kind(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    wt = self.make_branch_and_tree('path')
    wt.lock_write()
    self.addCleanup(wt.unlock)
    wt.commit('base', rev_id=b'A-id')
    os.symlink('bar', 'path/foo')
    wt.add(['foo'], ids=[b'foo-id'])
    wt.commit('add symlink foo => bar', rev_id=b'B-id')
    wt.set_last_revision(b'A-id')
    wt.branch.set_last_revision_info(1, b'A-id')
    wt.revert()
    wt.commit('C', rev_id=b'C-id')
    wt.merge_from_branch(wt.branch, b'B-id')
    self.assertEqual('bar', wt.get_symlink_target('foo'))
    os.remove('path/foo')
    os.symlink('bing', 'path/foo')
    wt.commit('E merges C & B, overrides to bing', rev_id=b'E-id')
    wt.set_last_revision(b'B-id')
    wt.branch.set_last_revision_info(2, b'B-id')
    wt.revert()
    wt.merge_from_branch(wt.branch, b'C-id')
    os.remove('path/foo')
    self.build_tree_contents([('path/foo', b'file content\n')])
    list(wt.iter_changes(wt.basis_tree()))
    wt.commit('D merges B & C, makes it a file', rev_id=b'D-id')
    merger = _mod_merge.Merger.from_revision_ids(wt, b'E-id')
    merger.merge_type = _mod_merge.Merge3Merger
    merge_obj = merger.make_merger()
    entries = list(merge_obj._entries_lca())
    root_id = wt.path2id('')
    self.assertEqual([(b'foo-id', True, ((None, ['foo', None]), 'foo', 'foo'), ((None, [root_id, None]), root_id, root_id), ((None, ['foo', None]), 'foo', 'foo'), ((None, [False, None]), False, False), False)], entries)