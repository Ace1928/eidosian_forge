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
def test_rename_fails(self):
    self.requireFeature(features.not_running_as_root)
    create, root_id = self.transform()
    create.new_directory('first-dir', root_id, b'first-id')
    create.new_file('myfile', root_id, [b'myfile-text'], b'myfile-id')
    create.apply()
    if os.name == 'posix' and sys.platform != 'cygwin':
        osutils.make_readonly(self.wt.abspath('first-dir'))
    elif os.name == 'nt':
        self.addCleanup(open(self.wt.abspath('myfile')).close)
    else:
        self.skipTest("Can't force a permissions error on rename")
    rename_transform, root_id = self.transform()
    file_trans_id = rename_transform.trans_id_tree_path('myfile')
    dir_id = rename_transform.trans_id_tree_path('first-dir')
    rename_transform.adjust_path('newname', dir_id, file_trans_id)
    e = self.assertRaises(TransformRenameFailed, rename_transform.apply)
    self.assertEqual(e.errno, errno.EACCES)
    if os.name == 'posix':
        self.assertEndsWith(e.to_path, '/first-dir/newname')
    else:
        self.assertEqual(os.path.basename(e.from_path), 'myfile')