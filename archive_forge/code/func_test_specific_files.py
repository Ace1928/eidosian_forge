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
def test_specific_files(self):
    revision_tree, preview_tree = self.get_tree_and_preview_tree()
    changes = preview_tree.iter_changes(revision_tree, specific_files=[''])
    a_entry = (('a', 'a'), True, (True, True), ('a', 'a'), ('file', 'file'), (False, False), False)
    self.assertThat(changes, MatchesTreeChanges(revision_tree, preview_tree, [a_entry]))