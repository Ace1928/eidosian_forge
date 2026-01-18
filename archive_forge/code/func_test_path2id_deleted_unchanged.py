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
def test_path2id_deleted_unchanged(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/unchanged', 'tree/deleted'])
    tree.add(['unchanged', 'deleted'])
    preview = tree.preview_transform()
    self.addCleanup(preview.finalize)
    preview.unversion_file(preview.trans_id_tree_path('deleted'))
    preview_tree = preview.get_preview_tree()
    self.assertEqual('unchanged', find_previous_path(preview_tree, tree, 'unchanged'))
    self.assertFalse(preview_tree.is_versioned('deleted'))