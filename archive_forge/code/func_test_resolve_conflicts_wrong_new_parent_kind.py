import errno
import os
import sys
import time
from io import BytesIO
from breezy.bzr.transform import resolve_checkout
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from ... import osutils, tests, trace, transform, urlutils
from ...bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ...errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ...osutils import file_kind, pathjoin
from ...transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ...transport import FileExists
from ...tree import TreeChange
from .. import TestSkipped, features
from ..features import HardlinkFeature, SymlinkFeature
def test_resolve_conflicts_wrong_new_parent_kind(self):
    tt, root = self.transform()
    parent_id = tt.new_directory('parent', root, b'parent-id')
    tt.new_file('child,', parent_id, [b'contents2'], b'file-id')
    tt.apply()
    tt, root = self.transform()
    parent_id = tt.trans_id_tree_path('parent')
    tt.delete_contents(parent_id)
    tt.create_file([b'contents'], parent_id)
    raw_conflicts = resolve_conflicts(tt)
    self.assertEqual({('non-directory parent', 'Created directory', 'new-3')}, raw_conflicts)
    tt.apply()
    if self.wt.has_versioned_directories():
        self.assertFalse(self.wt.is_versioned('parent'))
        self.assertTrue(self.wt.is_versioned('parent.new'))
    if self.wt.supports_setting_file_ids():
        self.assertEqual(b'parent-id', self.wt.path2id('parent.new'))