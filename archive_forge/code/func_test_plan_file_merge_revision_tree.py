import os
from io import BytesIO
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_tree import TestCaseWithTree
from ... import revision as _mod_revision
from ... import tests, trace
from ...diff import show_diff_trees
from ...merge import Merge3Merger, Merger
from ...transform import ROOT_PARENT, resolve_conflicts
from ...tree import TreeChange, find_previous_path
from ..features import SymlinkFeature, UnicodeFilenameFeature
def test_plan_file_merge_revision_tree(self):
    work_a = self.make_branch_and_tree('wta')
    self.build_tree_contents([('wta/file', b'a\nb\nc\nd\n')])
    work_a.add('file')
    base_id = work_a.commit('base version')
    tree_b = work_a.controldir.sprout('wtb').open_workingtree()
    preview = work_a.basis_tree().preview_transform()
    self.addCleanup(preview.finalize)
    trans_id = preview.trans_id_tree_path('file')
    preview.delete_contents(trans_id)
    preview.create_file([b'b\nc\nd\ne\n'], trans_id)
    self.build_tree_contents([('wtb/file', b'a\nc\nd\nf\n')])
    tree_a = preview.get_preview_tree()
    if not getattr(tree_a, 'plan_file_merge', None):
        self.skipTest('tree does not support file merge planning')
    tree_a.set_parent_ids([base_id])
    self.addCleanup(tree_b.lock_read().unlock)
    self.assertEqual([('killed-a', b'a\n'), ('killed-b', b'b\n'), ('unchanged', b'c\n'), ('unchanged', b'd\n'), ('new-a', b'e\n'), ('new-b', b'f\n')], list(tree_a.plan_file_merge('file', tree_b)))