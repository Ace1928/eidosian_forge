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
class TestFinalizeRobustness(tests.TestCaseWithTransport):
    """Ensure treetransform creation errors can be safely cleaned up after"""

    def _override_globals_in_method(self, instance, method_name, globals):
        """Replace method on instance with one with updated globals"""
        import types
        func = getattr(instance, method_name).__func__
        new_globals = dict(func.__globals__)
        new_globals.update(globals)
        new_func = types.FunctionType(func.__code__, new_globals, func.__name__, func.__defaults__)
        setattr(instance, method_name, types.MethodType(new_func, instance))
        self.addCleanup(delattr, instance, method_name)

    @staticmethod
    def _fake_open_raises_before(name, mode):
        """Like open() but raises before doing anything"""
        raise RuntimeError

    @staticmethod
    def _fake_open_raises_after(name, mode):
        """Like open() but raises after creating file without returning"""
        open(name, mode).close()
        raise RuntimeError

    def create_transform_and_root_trans_id(self):
        """Setup a transform creating a file in limbo"""
        tree = self.make_branch_and_tree('.')
        tt = tree.transform()
        return (tt, tt.create_path('a', tt.root))

    def create_transform_and_subdir_trans_id(self):
        """Setup a transform creating a directory containing a file in limbo"""
        tree = self.make_branch_and_tree('.')
        tt = tree.transform()
        d_trans_id = tt.create_path('d', tt.root)
        tt.create_directory(d_trans_id)
        f_trans_id = tt.create_path('a', d_trans_id)
        tt.adjust_path('a', d_trans_id, f_trans_id)
        return (tt, f_trans_id)

    def test_root_create_file_open_raises_before_creation(self):
        tt, trans_id = self.create_transform_and_root_trans_id()
        self._override_globals_in_method(tt, 'create_file', {'open': self._fake_open_raises_before})
        self.assertRaises(RuntimeError, tt.create_file, [b'contents'], trans_id)
        path = tt._limbo_name(trans_id)
        self.assertPathDoesNotExist(path)
        tt.finalize()
        self.assertPathDoesNotExist(tt._limbodir)

    def test_root_create_file_open_raises_after_creation(self):
        tt, trans_id = self.create_transform_and_root_trans_id()
        self._override_globals_in_method(tt, 'create_file', {'open': self._fake_open_raises_after})
        self.assertRaises(RuntimeError, tt.create_file, [b'contents'], trans_id)
        path = tt._limbo_name(trans_id)
        self.assertPathExists(path)
        tt.finalize()
        self.assertPathDoesNotExist(path)
        self.assertPathDoesNotExist(tt._limbodir)

    def test_subdir_create_file_open_raises_before_creation(self):
        tt, trans_id = self.create_transform_and_subdir_trans_id()
        self._override_globals_in_method(tt, 'create_file', {'open': self._fake_open_raises_before})
        self.assertRaises(RuntimeError, tt.create_file, [b'contents'], trans_id)
        path = tt._limbo_name(trans_id)
        self.assertPathDoesNotExist(path)
        tt.finalize()
        self.assertPathDoesNotExist(tt._limbodir)

    def test_subdir_create_file_open_raises_after_creation(self):
        tt, trans_id = self.create_transform_and_subdir_trans_id()
        self._override_globals_in_method(tt, 'create_file', {'open': self._fake_open_raises_after})
        self.assertRaises(RuntimeError, tt.create_file, [b'contents'], trans_id)
        path = tt._limbo_name(trans_id)
        self.assertPathExists(path)
        tt.finalize()
        self.assertPathDoesNotExist(path)
        self.assertPathDoesNotExist(tt._limbodir)

    def test_rename_in_limbo_rename_raises_after_rename(self):
        tt, trans_id = self.create_transform_and_root_trans_id()
        parent1 = tt.new_directory('parent1', tt.root)
        child1 = tt.new_file('child1', parent1, [b'contents'])
        parent2 = tt.new_directory('parent2', tt.root)

        class FakeOSModule:

            def rename(self, old, new):
                os.rename(old, new)
                raise RuntimeError
        self._override_globals_in_method(tt, '_rename_in_limbo', {'os': FakeOSModule()})
        self.assertRaises(RuntimeError, tt.adjust_path, 'child1', parent2, child1)
        path = osutils.pathjoin(tt._limbo_name(parent2), 'child1')
        self.assertPathExists(path)
        tt.finalize()
        self.assertPathDoesNotExist(path)
        self.assertPathDoesNotExist(tt._limbodir)

    def test_rename_in_limbo_rename_raises_before_rename(self):
        tt, trans_id = self.create_transform_and_root_trans_id()
        parent1 = tt.new_directory('parent1', tt.root)
        child1 = tt.new_file('child1', parent1, [b'contents'])
        parent2 = tt.new_directory('parent2', tt.root)

        class FakeOSModule:

            def rename(self, old, new):
                raise RuntimeError
        self._override_globals_in_method(tt, '_rename_in_limbo', {'os': FakeOSModule()})
        self.assertRaises(RuntimeError, tt.adjust_path, 'child1', parent2, child1)
        path = osutils.pathjoin(tt._limbo_name(parent1), 'child1')
        self.assertPathExists(path)
        tt.finalize()
        self.assertPathDoesNotExist(path)
        self.assertPathDoesNotExist(tt._limbodir)