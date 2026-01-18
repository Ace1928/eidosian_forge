import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_after_non_existant_non_ascii(self):
    """Check error renaming after move with missing non-ascii file"""
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    tree.add(['a'])
    e = self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'a', '§', after=True)
    self.assertIsInstance(e.extra, _mod_transport.NoSuchFile)
    self.assertEqual(e.extra.path, '§')