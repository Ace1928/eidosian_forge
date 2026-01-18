import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def test_parse_binary_after_normal(self):
    patches = list(parse_patches(self.data_lines('binary-after-normal.patch')))
    self.assertIs(BinaryPatch, patches[1].__class__)
    self.assertIs(Patch, patches[0].__class__)
    self.assertContainsRe(patches[1].oldname, b'^bar\t')
    self.assertContainsRe(patches[1].newname, b'^qux\t')
    self.assertContainsRe(patches[1].as_bytes(), b'Binary files bar\t.* and qux\t.* differ\n')