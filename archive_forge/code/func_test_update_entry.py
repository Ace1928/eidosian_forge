import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_update_entry(self):
    state, _ = self.get_state_with_a()
    tree = self.make_branch_and_tree('tree')
    tree.lock_write()
    empty_revid = tree.commit('empty')
    self.build_tree(['tree/a'])
    tree.add(['a'], ids=[b'a-id'])
    with_a_id = tree.commit('with_a')
    self.addCleanup(tree.unlock)
    state.set_parent_trees([(empty_revid, tree.branch.repository.revision_tree(empty_revid))], [])
    entry = state._get_entry(0, path_utf8=b'a')
    self.build_tree(['a'])
    self.assertEqual((b'', b'a', b'a-id'), entry[0])
    self.assertEqual((b'f', b'', 0, False, dirstate.DirState.NULLSTAT), entry[1][0])
    state.save()
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
    stat_value = os.lstat('a')
    packed_stat = dirstate.pack_stat(stat_value)
    link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
    self.assertEqual(None, link_or_sha1)
    self.assertEqual((b'f', b'', 14, False, dirstate.DirState.NULLSTAT), entry[1][0])
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
    mode = stat_value.st_mode
    self.assertEqual([('is_exec', mode, False)], state._log)
    state.save()
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
    state.adjust_time(-10)
    del state._log[:]
    link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
    self.assertEqual([('is_exec', mode, False)], state._log)
    self.assertEqual(None, link_or_sha1)
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
    self.assertEqual((b'f', b'', 14, False, dirstate.DirState.NULLSTAT), entry[1][0])
    state.save()
    state.adjust_time(+20)
    del state._log[:]
    link_or_sha1 = dirstate.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
    self.assertEqual(None, link_or_sha1)
    self.assertEqual([('is_exec', mode, False)], state._log)
    self.assertEqual((b'f', b'', 14, False, dirstate.DirState.NULLSTAT), entry[1][0])
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
    del state._log[:]
    state.set_parent_trees([(with_a_id, tree.branch.repository.revision_tree(with_a_id))], [])
    entry = state._get_entry(0, path_utf8=b'a')
    link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
    self.assertEqual(b'b50e5406bb5e153ebbeb20268fcf37c87e1ecfb6', link_or_sha1)
    self.assertEqual([('is_exec', mode, False), ('sha1', b'a')], state._log)
    self.assertEqual((b'f', link_or_sha1, 14, False, packed_stat), entry[1][0])
    del state._log[:]
    link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
    self.assertEqual(b'b50e5406bb5e153ebbeb20268fcf37c87e1ecfb6', link_or_sha1)
    self.assertEqual([], state._log)
    self.assertEqual((b'f', link_or_sha1, 14, False, packed_stat), entry[1][0])