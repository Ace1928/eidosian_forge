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
def test_path2id_created(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/unchanged'])
    tree.add(['unchanged'])
    preview = tree.preview_transform()
    self.addCleanup(preview.finalize)
    preview.new_file('new', preview.trans_id_tree_path('unchanged'), [b'contents'], b'new-id')
    preview_tree = preview.get_preview_tree()
    self.assertTrue(preview_tree.is_versioned('unchanged/new'))
    if self.workingtree_format.supports_setting_file_ids:
        self.assertEqual(b'new-id', preview_tree.path2id('unchanged/new'))