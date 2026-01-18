import os
from io import BytesIO
import breezy
from .. import config, controldir, errors, trace
from .. import transport as _mod_transport
from ..branch import Branch
from ..bzr.bzrdir import BzrDirMetaFormat1
from ..commit import (CannotCommitSelectedFileMerge, Commit,
from ..errors import BzrError, LockContention
from ..tree import TreeChange
from . import TestCase, TestCaseWithTransport, test_foreign
from .features import SymlinkFeature
from .matchers import MatchesAncestry, MatchesTreeChanges
def test_safe_master_lock(self):
    os.mkdir('master')
    master = BzrDirMetaFormat1().initialize('master')
    master.create_repository()
    master_branch = master.create_branch()
    master.create_workingtree()
    bound = master.sprout('bound')
    wt = bound.open_workingtree()
    wt.branch.set_bound_location(os.path.realpath('master'))
    with master_branch.lock_write():
        self.assertRaises(LockContention, wt.commit, 'silly')