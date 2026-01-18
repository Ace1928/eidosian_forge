import codecs
import errno
import os
import sys
import time
from io import BytesIO, StringIO
import fastbencode as bencode
from .. import filters, osutils
from .. import revision as _mod_revision
from .. import rules, tests, trace, transform, urlutils
from ..bzr import generate_ids
from ..bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ..controldir import ControlDir
from ..diff import show_diff_trees
from ..errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ..merge import Merge3Merger, Merger
from ..mutabletree import MutableTree
from ..osutils import file_kind, pathjoin
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..transport import FileExists
from . import TestCaseInTempDir, TestSkipped, features
from .features import HardlinkFeature, SymlinkFeature
def test_text_merge(self):
    root_id = generate_ids.gen_root_id()
    base = TransformGroup('base', root_id)
    base.tt.new_file('a', base.root, [b'a\nb\nc\nd\x08e\n'], b'a')
    base.tt.new_file('b', base.root, [b'b1'], b'b')
    base.tt.new_file('c', base.root, [b'c'], b'c')
    base.tt.new_file('d', base.root, [b'd'], b'd')
    base.tt.new_file('e', base.root, [b'e'], b'e')
    base.tt.new_file('f', base.root, [b'f'], b'f')
    base.tt.new_directory('g', base.root, b'g')
    base.tt.new_directory('h', base.root, b'h')
    base.tt.apply()
    other = TransformGroup('other', root_id)
    other.tt.new_file('a', other.root, [b'y\nb\nc\nd\x08e\n'], b'a')
    other.tt.new_file('b', other.root, [b'b2'], b'b')
    other.tt.new_file('c', other.root, [b'c2'], b'c')
    other.tt.new_file('d', other.root, [b'd'], b'd')
    other.tt.new_file('e', other.root, [b'e2'], b'e')
    other.tt.new_file('f', other.root, [b'f'], b'f')
    other.tt.new_file('g', other.root, [b'g'], b'g')
    other.tt.new_file('h', other.root, [b'h\ni\nj\nk\n'], b'h')
    other.tt.new_file('i', other.root, [b'h\ni\nj\nk\n'], b'i')
    other.tt.apply()
    this = TransformGroup('this', root_id)
    this.tt.new_file('a', this.root, [b'a\nb\nc\nd\x08z\n'], b'a')
    this.tt.new_file('b', this.root, [b'b'], b'b')
    this.tt.new_file('c', this.root, [b'c'], b'c')
    this.tt.new_file('d', this.root, [b'd2'], b'd')
    this.tt.new_file('e', this.root, [b'e2'], b'e')
    this.tt.new_file('f', this.root, [b'f'], b'f')
    this.tt.new_file('g', this.root, [b'g'], b'g')
    this.tt.new_file('h', this.root, [b'1\n2\n3\n4\n'], b'h')
    this.tt.new_file('i', this.root, [b'1\n2\n3\n4\n'], b'i')
    this.tt.apply()
    Merge3Merger(this.wt, this.wt, base.wt, other.wt)
    with this.wt.get_file('a') as f:
        self.assertEqual(f.read(), b'y\nb\nc\nd\x08z\n')
    with this.wt.get_file('b') as f:
        self.assertEqual(f.read(), conflict_text(b'b', b'b2'))
    self.assertEqual(this.wt.get_file('c').read(), b'c2')
    self.assertEqual(this.wt.get_file('d').read(), b'd2')
    self.assertEqual(this.wt.get_file('e').read(), b'e2')
    self.assertEqual(this.wt.get_file('f').read(), b'f')
    self.assertEqual(this.wt.get_file('g').read(), b'g')
    self.assertEqual(this.wt.get_file('h').read(), conflict_text(b'1\n2\n3\n4\n', b'h\ni\nj\nk\n'))
    self.assertEqual(this.wt.get_file('h.THIS').read(), b'1\n2\n3\n4\n')
    self.assertEqual(this.wt.get_file('h.OTHER').read(), b'h\ni\nj\nk\n')
    self.assertEqual(file_kind(this.wt.abspath('h.BASE')), 'directory')
    self.assertEqual(this.wt.get_file('i').read(), conflict_text(b'1\n2\n3\n4\n', b'h\ni\nj\nk\n'))
    self.assertEqual(this.wt.get_file('i.THIS').read(), b'1\n2\n3\n4\n')
    self.assertEqual(this.wt.get_file('i.OTHER').read(), b'h\ni\nj\nk\n')
    self.assertEqual(os.path.exists(this.wt.abspath('i.BASE')), False)
    modified = ['a', 'b', 'c', 'h', 'i']
    merge_modified = this.wt.merge_modified()
    self.assertSubset(merge_modified, modified)
    self.assertEqual(len(merge_modified), len(modified))
    with open(this.wt.abspath('a'), 'wb') as f:
        f.write(b'booga')
    modified.pop(0)
    merge_modified = this.wt.merge_modified()
    self.assertSubset(merge_modified, modified)
    self.assertEqual(len(merge_modified), len(modified))
    this.wt.remove('b')
    this.wt.revert()