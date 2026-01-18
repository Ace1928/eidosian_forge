import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_symlink_conflict_handling(self):
    """Ensure that when building trees, conflict handling is done"""
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    source = self.make_branch_and_tree('source')
    os.symlink('foo', 'source/symlink')
    source.add('symlink', ids=b'new-symlink')
    source.commit('added file')
    target = self.make_branch_and_tree('target')
    os.symlink('bar', 'target/symlink')
    build_tree(source.basis_tree(), target)
    self.assertEqual([DuplicateEntry('Moved existing file to', 'symlink.moved', 'symlink', None, 'new-symlink')], target.conflicts())
    target = self.make_branch_and_tree('target2')
    os.symlink('foo', 'target2/symlink')
    build_tree(source.basis_tree(), target)
    self.assertEqual([], target.conflicts())