import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_observed_sha1_new_file(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['foo'])
    tree.add(['foo'], ids=[b'foo-id'])
    with tree.lock_read():
        current_sha1 = tree._get_entry(path='foo')[1][0][1]
    with tree.lock_write():
        tree._observed_sha1('foo', (osutils.sha_file_by_name('foo'), os.lstat('foo')))
        self.assertEqual(current_sha1, tree._get_entry(path='foo')[1][0][1])