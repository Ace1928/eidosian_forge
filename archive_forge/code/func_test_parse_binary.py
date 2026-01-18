import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def test_parse_binary(self):
    """Test parsing a whole patch"""
    patches = list(parse_patches(self.data_lines('binary.patch')))
    self.assertIs(BinaryPatch, patches[0].__class__)
    self.assertIs(Patch, patches[1].__class__)
    self.assertEqual(patches[0].oldname, b'bar')
    self.assertEqual(patches[0].newname, b'qux')
    self.assertContainsRe(patches[0].as_bytes(), b'Binary files bar and qux differ\n')