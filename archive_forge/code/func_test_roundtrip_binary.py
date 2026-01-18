import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def test_roundtrip_binary(self):
    patchtext = b''.join(self.data_lines('binary.patch'))
    patches = parse_patches(patchtext.splitlines(True))
    self.assertEqual(patchtext, b''.join((p.as_bytes() for p in patches)))