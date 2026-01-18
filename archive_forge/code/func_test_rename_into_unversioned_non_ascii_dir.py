import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_into_unversioned_non_ascii_dir(self):
    """Check error when renaming into unversioned non-ascii directory"""
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', '§/'])
    tree.add(['a'])
    if tree.has_versioned_directories():
        e = self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'a', '§/a')
        self.assertIsInstance(e.extra, errors.NotVersionedError)
        self.assertEqual(e.extra.path, '§')
    else:
        tree.rename_one('a', '§/a')