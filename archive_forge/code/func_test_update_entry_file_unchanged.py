import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_update_entry_file_unchanged(self):
    state, _ = self.get_state_with_a()
    tree = self.make_branch_and_tree('tree')
    tree.lock_write()
    self.build_tree(['tree/a'])
    tree.add(['a'], ids=[b'a-id'])
    with_a_id = tree.commit('witha')
    self.addCleanup(tree.unlock)
    state.set_parent_trees([(with_a_id, tree.branch.repository.revision_tree(with_a_id))], [])
    entry = state._get_entry(0, path_utf8=b'a')
    self.build_tree(['a'])
    sha1sum = b'b50e5406bb5e153ebbeb20268fcf37c87e1ecfb6'
    state.adjust_time(+20)
    self.assertEqual(sha1sum, self.do_update_entry(state, entry, b'a'))
    self.assertEqual(dirstate.DirState.IN_MEMORY_MODIFIED, state._dirblock_state)
    state.save()
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
    self.assertEqual(sha1sum, self.do_update_entry(state, entry, b'a'))
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)