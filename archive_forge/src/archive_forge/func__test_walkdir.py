import os
from breezy.tests.features import SymlinkFeature
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def _test_walkdir(self, file_status, prefix=''):
    result = []
    tree, expected_dirblocks = self.get_tree(file_status, prefix)
    with tree.lock_read():
        for dirpath, dirblock in tree.walkdirs(prefix):
            result.append((dirpath, list(dirblock)))
    for pos, item in enumerate(expected_dirblocks):
        result_pos = []
        if len(result) > pos:
            result_pos = result[pos]
        self.assertEqual(item, result_pos)
    self.assertEqual(expected_dirblocks, result)