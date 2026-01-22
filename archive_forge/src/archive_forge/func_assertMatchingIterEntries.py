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
def assertMatchingIterEntries(self, tt, specific_files=None):
    preview_tree = tt.get_preview_tree()
    preview_result = list(preview_tree.iter_entries_by_dir(specific_files=specific_files))
    tree = tt._tree
    tt.apply()
    actual_result = list(tree.iter_entries_by_dir(specific_files=specific_files))
    self.assertEqual(actual_result, preview_result)