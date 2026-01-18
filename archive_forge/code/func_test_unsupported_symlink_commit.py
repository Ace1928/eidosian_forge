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
def test_unsupported_symlink_commit(self):
    self.requireFeature(SymlinkFeature(self.test_dir))
    tree = self.make_branch_and_tree('.')
    self.build_tree(['hello'])
    tree.add('hello')
    tree.commit('added hello', rev_id=b'hello_id')
    os.symlink('hello', 'foo')
    tree.add('foo')
    tree.commit('added foo', rev_id=b'foo_id')
    log = BytesIO()
    trace.push_log_file(log)
    os_symlink = getattr(os, 'symlink', None)
    os.symlink = None
    try:
        os.unlink('foo')
        self.build_tree(['world'])
        tree.add('world')
        tree.commit('added world', rev_id=b'world_id')
    finally:
        if os_symlink:
            os.symlink = os_symlink
    self.assertContainsRe(log.getvalue(), b'Ignoring "foo" as symlinks are not supported on this filesystem\\.')