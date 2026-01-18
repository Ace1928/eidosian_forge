import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def test_preserve_dirty_head(self):
    """Parse a patch containing a dirty header, and preserve lines"""
    lines = [b"=== added directory 'foo/bar'\n", b"=== modified file 'orig/commands.py'\n", b'--- orig/commands.py\n', b'+++ mod/dommands.py\n', b"=== modified file 'orig/another.py'\n", b'--- orig/another.py\n', b'+++ mod/another.py\n']
    patches = list(parse_patches(lines.__iter__(), allow_dirty=True, keep_dirty=True))
    self.assertLength(2, patches)
    self.assertEqual(patches[0]['dirty_head'], [b"=== added directory 'foo/bar'\n", b"=== modified file 'orig/commands.py'\n"])
    self.assertEqual(patches[0]['patch'].get_header().splitlines(True), [b'--- orig/commands.py\n', b'+++ mod/dommands.py\n'])
    self.assertEqual(patches[1]['dirty_head'], [b"=== modified file 'orig/another.py'\n"])
    self.assertEqual(patches[1]['patch'].get_header().splitlines(True), [b'--- orig/another.py\n', b'+++ mod/another.py\n'])