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
def _test_symlinks(self, link_name1, link_target1, link_name2, link_target2):

    def ozpath(p):
        return 'oz/' + p
    self.requireFeature(SymlinkFeature(self.test_dir))
    transform, root = self.transform()
    oz_id = transform.new_directory('oz', root, b'oz-id')
    transform.new_symlink(link_name1, oz_id, link_target1, b'wizard-id')
    wiz_id = transform.create_path(link_name2, oz_id)
    transform.create_symlink(link_target2, wiz_id)
    transform.version_file(wiz_id, file_id=b'wiz-id2')
    transform.set_executability(True, wiz_id)
    self.assertEqual(transform.find_raw_conflicts(), [('non-file executability', wiz_id)])
    transform.set_executability(None, wiz_id)
    transform.apply()
    if self.wt.supports_setting_file_ids():
        self.assertEqual(self.wt.path2id(ozpath(link_name1)), b'wizard-id')
    self.assertEqual('symlink', file_kind(self.wt.abspath(ozpath(link_name1))))
    self.assertEqual(link_target2, osutils.readlink(self.wt.abspath(ozpath(link_name2))))
    self.assertEqual(link_target1, osutils.readlink(self.wt.abspath(ozpath(link_name1))))