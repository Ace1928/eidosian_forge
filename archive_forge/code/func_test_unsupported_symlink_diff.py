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
def test_unsupported_symlink_diff(self):
    self.requireFeature(SymlinkFeature(self.test_dir))
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('a', 'content 1')])
    tree.add('a')
    os.symlink('a', 'foo')
    tree.add('foo')
    revid1 = tree.commit('rev1')
    revision_tree = tree.branch.repository.revision_tree(revid1)
    os_symlink = getattr(os, 'symlink', None)
    os.symlink = None
    try:
        preview = revision_tree.preview_transform()
        self.addCleanup(preview.finalize)
        preview.delete_versioned(preview.trans_id_tree_path('foo'))
        preview_tree = preview.get_preview_tree()
        out = BytesIO()
        log = BytesIO()
        trace.push_log_file(log)
        show_diff_trees(revision_tree, preview_tree, out)
        lines = out.getvalue().splitlines()
    finally:
        os.symlink = os_symlink
    self.assertContainsRe(log.getvalue(), b'Ignoring "foo" as symlinks are not supported on this filesystem')