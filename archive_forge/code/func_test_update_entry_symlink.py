import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_update_entry_symlink(self):
    """Update entry should read symlinks."""
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    state, entry = self.get_state_with_a()
    state.save()
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
    os.symlink('target', 'a')
    state.adjust_time(-10)
    stat_value = os.lstat('a')
    packed_stat = dirstate.pack_stat(stat_value)
    link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
    self.assertEqual(b'target', link_or_sha1)
    self.assertEqual([('read_link', b'a', b'')], state._log)
    self.assertEqual([(b'l', b'', 6, False, dirstate.DirState.NULLSTAT)], entry[1])
    self.assertEqual(dirstate.DirState.IN_MEMORY_HASH_MODIFIED, state._dirblock_state)
    del state._log[:]
    link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
    self.assertEqual(b'target', link_or_sha1)
    self.assertEqual([('read_link', b'a', b'')], state._log)
    self.assertEqual([(b'l', b'', 6, False, dirstate.DirState.NULLSTAT)], entry[1])
    state.save()
    state.adjust_time(+20)
    del state._log[:]
    link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
    self.assertEqual(b'target', link_or_sha1)
    self.assertEqual([('read_link', b'a', b'')], state._log)
    self.assertEqual([(b'l', b'target', 6, False, packed_stat)], entry[1])
    del state._log[:]
    self.assertEqual([], state._log)
    link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
    self.assertEqual(b'target', link_or_sha1)
    self.assertEqual([(b'l', b'target', 6, False, packed_stat)], entry[1])