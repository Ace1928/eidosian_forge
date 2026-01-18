import os
import sys
import time
from breezy import tests
from breezy.bzr import hashcache
from breezy.bzr.workingtree import InventoryWorkingTree
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
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