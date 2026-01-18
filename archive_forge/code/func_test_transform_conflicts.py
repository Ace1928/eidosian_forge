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
def test_transform_conflicts(self):
    revision_tree = self.create_tree()
    preview = revision_tree.preview_transform()
    self.addCleanup(preview.finalize)
    preview.new_file('a', preview.root, [b'content 2'])
    resolve_conflicts(preview)
    trans_id = preview.trans_id_tree_path('a')
    self.assertEqual('a.moved', preview.final_name(trans_id))