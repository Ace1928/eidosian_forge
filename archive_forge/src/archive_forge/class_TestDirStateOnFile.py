import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
class TestDirStateOnFile(TestCaseWithDirState):

    def create_updated_dirstate(self):
        self.build_tree(['a-file'])
        tree = self.make_branch_and_tree('.')
        tree.add(['a-file'], ids=[b'a-id'])
        tree.commit('add a-file')
        state = dirstate.DirState.from_tree(tree, 'dirstate')
        state.save()
        state.unlock()
        state = dirstate.DirState.on_file('dirstate')
        state.lock_read()
        return state

    def test_construct_with_path(self):
        tree = self.make_branch_and_tree('tree')
        state = dirstate.DirState.from_tree(tree, 'dirstate.from_tree')
        lines = state.get_lines()
        state.unlock()
        self.build_tree_contents([('dirstate', b''.join(lines))])
        expected_result = ([], [((b'', b'', tree.path2id('')), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT)])])
        state = dirstate.DirState.on_file('dirstate')
        state.lock_write()
        self.check_state_with_reopen(expected_result, state)

    def test_can_save_clean_on_file(self):
        tree = self.make_branch_and_tree('tree')
        state = dirstate.DirState.from_tree(tree, 'dirstate')
        try:
            state.save()
        finally:
            state.unlock()

    def test_can_save_in_read_lock(self):
        state = self.create_updated_dirstate()
        try:
            entry = state._get_entry(0, path_utf8=b'a-file')
            self.assertEqual(0, entry[1][0][2])
            self.assertNotEqual((None, None), entry)
            state._sha_cutoff_time()
            state._cutoff_time += 10.0
            st = os.lstat('a-file')
            sha1sum = dirstate.update_entry(state, entry, 'a-file', st)
            self.assertEqual(b'ecc5374e9ed82ad3ea3b4d452ea995a5fd3e70e3', sha1sum)
            self.assertEqual(st.st_size, entry[1][0][2])
            self.assertEqual(dirstate.DirState.IN_MEMORY_HASH_MODIFIED, state._dirblock_state)
            del entry
            state.save()
        finally:
            state.unlock()
        state = dirstate.DirState.on_file('dirstate')
        state.lock_read()
        try:
            entry = state._get_entry(0, path_utf8=b'a-file')
            self.assertEqual(st.st_size, entry[1][0][2])
        finally:
            state.unlock()

    def test_save_fails_quietly_if_locked(self):
        """If dirstate is locked, save will fail without complaining."""
        state = self.create_updated_dirstate()
        try:
            entry = state._get_entry(0, path_utf8=b'a-file')
            self.assertEqual(b'', entry[1][0][1])
            state._sha_cutoff_time()
            state._cutoff_time += 10.0
            st = os.lstat('a-file')
            sha1sum = dirstate.update_entry(state, entry, 'a-file', st)
            self.assertEqual(b'ecc5374e9ed82ad3ea3b4d452ea995a5fd3e70e3', sha1sum)
            self.assertEqual(dirstate.DirState.IN_MEMORY_HASH_MODIFIED, state._dirblock_state)
            state2 = dirstate.DirState.on_file('dirstate')
            state2.lock_read()
            try:
                state.save()
            finally:
                state2.unlock()
        finally:
            state.unlock()
        state = dirstate.DirState.on_file('dirstate')
        state.lock_read()
        try:
            entry = state._get_entry(0, path_utf8=b'a-file')
            self.assertEqual(b'', entry[1][0][1])
        finally:
            state.unlock()

    def test_save_refuses_if_changes_aborted(self):
        self.build_tree(['a-file', 'a-dir/'])
        state = dirstate.DirState.initialize('dirstate')
        try:
            state.add('a-file', b'a-file-id', 'file', None, b'')
            state.save()
        finally:
            state.unlock()
        expected_blocks = [(b'', [((b'', b'', b'TREE_ROOT'), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT)])]), (b'', [((b'', b'a-file', b'a-file-id'), [(b'f', b'', 0, False, dirstate.DirState.NULLSTAT)])])]
        state = dirstate.DirState.on_file('dirstate')
        state.lock_write()
        try:
            state._read_dirblocks_if_needed()
            self.assertEqual(expected_blocks, state._dirblocks)
            state.add('a-dir', b'a-dir-id', 'directory', None, b'')
            state._changes_aborted = True
            state.save()
        finally:
            state.unlock()
        state = dirstate.DirState.on_file('dirstate')
        state.lock_read()
        try:
            state._read_dirblocks_if_needed()
            self.assertEqual(expected_blocks, state._dirblocks)
        finally:
            state.unlock()