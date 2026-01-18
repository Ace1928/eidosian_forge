from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_reordered(self):
    """Check for a corner case that requires re-starting the cursor"""
    vf = multiparent.MultiMemoryVersionedFile()
    self.add_version(vf, b'c', b'rev-a', [])
    self.add_version(vf, b'acb', b'rev-b', [b'rev-a'])
    self.add_version(vf, b'b', b'rev-c', [b'rev-b'])
    self.add_version(vf, b'a', b'rev-d', [b'rev-b'])
    self.add_version(vf, b'ba', b'rev-e', [b'rev-c', b'rev-d'])
    vf.clear_cache()
    lines = vf.get_line_list([b'rev-e'])[0]
    self.assertEqual([b'b\n', b'a\n'], lines)