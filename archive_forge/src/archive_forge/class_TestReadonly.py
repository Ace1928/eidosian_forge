import os
import sys
import time
from breezy import tests
from breezy.bzr import hashcache
from breezy.bzr.workingtree import InventoryWorkingTree
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
class TestReadonly(TestCaseWithWorkingTree):

    def setUp(self):
        if not self.platform_supports_readonly_dirs():
            raise tests.TestSkipped('platform does not support readonly directories.')
        super().setUp()

    def platform_supports_readonly_dirs(self):
        if sys.platform in ('win32', 'cygwin'):
            return False
        return True

    def _set_all_dirs(self, basedir, readonly=True):
        """Recursively set all directories beneath this one."""
        if readonly:
            mode = 365
        else:
            mode = 493
        for root, dirs, files in os.walk(basedir, topdown=False):
            for d in dirs:
                path = os.path.join(root, d)
                os.chmod(path, mode)

    def set_dirs_readonly(self, basedir):
        """Set all directories readonly, and have it cleanup on test exit."""
        self.addCleanup(self._set_all_dirs, basedir, readonly=False)
        self._set_all_dirs(basedir, readonly=True)

    def create_basic_tree(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/a', 'tree/b/', 'tree/b/c'])
        tree.add(['a', 'b', 'b/c'])
        tree.commit('creating an initial tree.')
        return tree

    def _custom_cutoff_time(self):
        """We need to fake the cutoff time."""
        return time.time() + 10.0

    def test_readonly_unclean(self):
        """Even if the tree is unclean, we should still handle readonly dirs."""
        tree = self.create_basic_tree()
        if not isinstance(tree, InventoryWorkingTree):
            raise tests.TestNotApplicable('requires inventory working tree')
        the_hashcache = getattr(tree, '_hashcache', None)
        if the_hashcache is not None:
            self.assertIsInstance(the_hashcache, hashcache.HashCache)
            the_hashcache._cutoff_time = self._custom_cutoff_time
            hack_dirstate = False
        else:
            hack_dirstate = True
        self.build_tree_contents([('tree/a', b'new contents of a\n')])
        self.set_dirs_readonly('tree')
        with tree.lock_read():
            if hack_dirstate:
                tree._dirstate._cutoff_time = self._custom_cutoff_time()
            for path in tree.all_versioned_paths():
                size = tree.get_file_size(path)
                sha1 = tree.get_file_sha1(path)