import os
from io import BytesIO
from .. import (conflicts, errors, symbol_versioning, trace, transport,
from ..bzr import bzrdir
from ..bzr import conflicts as _mod_bzr_conflicts
from ..bzr import workingtree as bzrworkingtree
from ..bzr import workingtree_3, workingtree_4
from ..lock import write_locked
from ..lockdir import LockDir
from ..tree import TreeDirectory, TreeEntry, TreeFile, TreeLink
from . import TestCase, TestCaseWithTransport, TestSkipped
from .features import SymlinkFeature
def test_unsupported_symlink_auto_resolve(self):
    self.requireFeature(SymlinkFeature(self.test_dir))
    base = self.make_branch_and_tree('base')
    self.build_tree_contents([('base/hello', 'Hello')])
    base.add('hello', ids=b'hello_id')
    base.commit('commit 0')
    other = base.controldir.sprout('other').open_workingtree()
    self.build_tree_contents([('other/hello', 'Hello')])
    os.symlink('other/hello', 'other/foo')
    other.add('foo', ids=b'foo_id')
    other.commit('commit symlink')
    this = base.controldir.sprout('this').open_workingtree()
    self.assertPathExists('this/hello')
    self.build_tree_contents([('this/hello', 'Hello')])
    this.commit('commit 2')
    log = BytesIO()
    trace.push_log_file(log)
    os_symlink = getattr(os, 'symlink', None)
    os.symlink = None
    try:
        this.merge_from_branch(other.branch)
    finally:
        if os_symlink:
            os.symlink = os_symlink
    self.assertContainsRe(log.getvalue(), b'Unable to create symlink "foo" on this filesystem')