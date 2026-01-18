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
def test_resolve_conflicts_wrong_existing_parent_kind(self):
    tt = self.prepare_wrong_parent_kind()
    raw_conflicts = resolve_conflicts(tt)
    self.assertEqual({('non-directory parent', 'Created directory', 'new-3')}, raw_conflicts)
    cooked_conflicts = list(tt.cook_conflicts(raw_conflicts))
    from ...bzr.workingtree import InventoryWorkingTree
    if isinstance(tt._tree, InventoryWorkingTree):
        self.assertEqual([NonDirectoryParent('Created directory', 'parent.new', b'parent-id')], cooked_conflicts)
    else:
        self.assertEqual(1, len(cooked_conflicts))
        self.assertEqual('parent.new', cooked_conflicts[0].path)
    tt.apply()
    if self.wt.has_versioned_directories():
        self.assertFalse(self.wt.is_versioned('parent'))
    if self.wt.supports_setting_file_ids():
        self.assertEqual(b'parent-id', self.wt.path2id('parent.new'))