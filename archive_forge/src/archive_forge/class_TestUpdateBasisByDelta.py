import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
class TestUpdateBasisByDelta(tests.TestCase):

    def path_to_ie(self, path, file_id, rev_id, dir_ids):
        if path.endswith('/'):
            is_dir = True
            path = path[:-1]
        else:
            is_dir = False
        dirname, basename = osutils.split(path)
        try:
            dir_id = dir_ids[dirname]
        except KeyError:
            dir_id = osutils.basename(dirname).encode('utf-8') + b'-id'
        if is_dir:
            ie = inventory.InventoryDirectory(file_id, basename, dir_id)
            dir_ids[path] = file_id
        else:
            ie = inventory.InventoryFile(file_id, basename, dir_id)
            ie.text_size = 0
            ie.text_sha1 = b''
        ie.revision = rev_id
        return ie

    def create_tree_from_shape(self, rev_id, shape):
        dir_ids = {'': b'root-id'}
        inv = inventory.Inventory(b'root-id', rev_id)
        for info in shape:
            if len(info) == 2:
                path, file_id = info
                ie_rev_id = rev_id
            else:
                path, file_id, ie_rev_id = info
            if path == '':
                del inv._byid[inv.root.file_id]
                inv.root.file_id = file_id
                inv._byid[file_id] = inv.root
                dir_ids[''] = file_id
                continue
            inv.add(self.path_to_ie(path, file_id, ie_rev_id, dir_ids))
        return inventorytree.InventoryRevisionTree(_Repo(), inv, rev_id)

    def create_empty_dirstate(self):
        fd, path = tempfile.mkstemp(prefix='bzr-dirstate')
        self.addCleanup(os.remove, path)
        os.close(fd)
        state = dirstate.DirState.initialize(path)
        self.addCleanup(state.unlock)
        return state

    def create_inv_delta(self, delta, rev_id):
        """Translate a 'delta shape' into an actual InventoryDelta"""
        dir_ids = {'': b'root-id'}
        inv_delta = []
        for old_path, new_path, file_id in delta:
            if old_path is not None and old_path.endswith('/'):
                old_path = old_path[:-1]
            if new_path is None:
                inv_delta.append((old_path, None, file_id, None))
                continue
            ie = self.path_to_ie(new_path, file_id, rev_id, dir_ids)
            inv_delta.append((old_path, new_path, file_id, ie))
        return inv_delta

    def assertUpdate(self, active, basis, target):
        """Assert that update_basis_by_delta works how we want.

        Set up a DirState object with active_shape for tree 0, basis_shape for
        tree 1. Then apply the delta from basis_shape to target_shape,
        and assert that the DirState is still valid, and that its stored
        content matches the target_shape.
        """
        active_tree = self.create_tree_from_shape(b'active', active)
        basis_tree = self.create_tree_from_shape(b'basis', basis)
        target_tree = self.create_tree_from_shape(b'target', target)
        state = self.create_empty_dirstate()
        state.set_state_from_scratch(active_tree.root_inventory, [(b'basis', basis_tree)], [])
        delta = target_tree.root_inventory._make_delta(basis_tree.root_inventory)
        state.update_basis_by_delta(delta, b'target')
        state._validate()
        dirstate_tree = workingtree_4.DirStateRevisionTree(state, b'target', _Repo(), None)
        self.assertEqual([], list(dirstate_tree.iter_changes(target_tree)))
        state2 = self.create_empty_dirstate()
        state2.set_state_from_scratch(active_tree.root_inventory, [(b'target', target_tree)], [])
        self.assertEqual(state2._dirblocks, state._dirblocks)
        return state

    def assertBadDelta(self, active, basis, delta):
        """Test that we raise InconsistentDelta when appropriate.

        :param active: The active tree shape
        :param basis: The basis tree shape
        :param delta: A description of the delta to apply. Similar to the form
            for regular inventory deltas, but omitting the InventoryEntry.
            So adding a file is: (None, 'path', b'file-id')
            Adding a directory is: (None, 'path/', b'dir-id')
            Renaming a dir is: ('old/', 'new/', b'dir-id')
            etc.
        """
        active_tree = self.create_tree_from_shape(b'active', active)
        basis_tree = self.create_tree_from_shape(b'basis', basis)
        inv_delta = self.create_inv_delta(delta, b'target')
        state = self.create_empty_dirstate()
        state.set_state_from_scratch(active_tree.root_inventory, [(b'basis', basis_tree)], [])
        self.assertRaises(errors.InconsistentDelta, state.update_basis_by_delta, inv_delta, b'target')
        self.assertTrue(state._changes_aborted)

    def test_remove_file_matching_active_state(self):
        state = self.assertUpdate(active=[], basis=[('file', b'file-id')], target=[])

    def test_remove_file_present_in_active_state(self):
        state = self.assertUpdate(active=[('file', b'file-id')], basis=[('file', b'file-id')], target=[])

    def test_remove_file_present_elsewhere_in_active_state(self):
        state = self.assertUpdate(active=[('other-file', b'file-id')], basis=[('file', b'file-id')], target=[])

    def test_remove_file_active_state_has_diff_file(self):
        state = self.assertUpdate(active=[('file', b'file-id-2')], basis=[('file', b'file-id')], target=[])

    def test_remove_file_active_state_has_diff_file_and_file_elsewhere(self):
        state = self.assertUpdate(active=[('file', b'file-id-2'), ('other-file', b'file-id')], basis=[('file', b'file-id')], target=[])

    def test_add_file_matching_active_state(self):
        state = self.assertUpdate(active=[('file', b'file-id')], basis=[], target=[('file', b'file-id')])

    def test_add_file_in_empty_dir_not_matching_active_state(self):
        state = self.assertUpdate(active=[], basis=[('dir/', b'dir-id')], target=[('dir/', b'dir-id', b'basis'), ('dir/file', b'file-id')])

    def test_add_file_missing_in_active_state(self):
        state = self.assertUpdate(active=[], basis=[], target=[('file', b'file-id')])

    def test_add_file_elsewhere_in_active_state(self):
        state = self.assertUpdate(active=[('other-file', b'file-id')], basis=[], target=[('file', b'file-id')])

    def test_add_file_active_state_has_diff_file_and_file_elsewhere(self):
        state = self.assertUpdate(active=[('other-file', b'file-id'), ('file', b'file-id-2')], basis=[], target=[('file', b'file-id')])

    def test_rename_file_matching_active_state(self):
        state = self.assertUpdate(active=[('other-file', b'file-id')], basis=[('file', b'file-id')], target=[('other-file', b'file-id')])

    def test_rename_file_missing_in_active_state(self):
        state = self.assertUpdate(active=[], basis=[('file', b'file-id')], target=[('other-file', b'file-id')])

    def test_rename_file_present_elsewhere_in_active_state(self):
        state = self.assertUpdate(active=[('third', b'file-id')], basis=[('file', b'file-id')], target=[('other-file', b'file-id')])

    def test_rename_file_active_state_has_diff_source_file(self):
        state = self.assertUpdate(active=[('file', b'file-id-2')], basis=[('file', b'file-id')], target=[('other-file', b'file-id')])

    def test_rename_file_active_state_has_diff_target_file(self):
        state = self.assertUpdate(active=[('other-file', b'file-id-2')], basis=[('file', b'file-id')], target=[('other-file', b'file-id')])

    def test_rename_file_active_has_swapped_files(self):
        state = self.assertUpdate(active=[('file', b'file-id'), ('other-file', b'file-id-2')], basis=[('file', b'file-id'), ('other-file', b'file-id-2')], target=[('file', b'file-id-2'), ('other-file', b'file-id')])

    def test_rename_file_basis_has_swapped_files(self):
        state = self.assertUpdate(active=[('file', b'file-id'), ('other-file', b'file-id-2')], basis=[('file', b'file-id-2'), ('other-file', b'file-id')], target=[('file', b'file-id'), ('other-file', b'file-id-2')])

    def test_rename_directory_with_contents(self):
        state = self.assertUpdate(active=[('dir1/', b'dir-id'), ('dir1/file', b'file-id')], basis=[('dir1/', b'dir-id'), ('dir1/file', b'file-id')], target=[('dir2/', b'dir-id'), ('dir2/file', b'file-id')])
        state = self.assertUpdate(active=[('dir2/', b'dir-id'), ('dir2/file', b'file-id')], basis=[('dir1/', b'dir-id'), ('dir1/file', b'file-id')], target=[('dir2/', b'dir-id'), ('dir2/file', b'file-id')])
        state = self.assertUpdate(active=[], basis=[('dir1/', b'dir-id'), ('dir1/file', b'file-id')], target=[('dir2/', b'dir-id'), ('dir2/file', b'file-id')])
        state = self.assertUpdate(active=[('dir3/', b'dir-id'), ('dir3/file', b'file-id')], basis=[('dir1/', b'dir-id'), ('dir1/file', b'file-id')], target=[('dir2/', b'dir-id'), ('dir2/file', b'file-id')])
        state = self.assertUpdate(active=[('dir1/', b'dir1-id'), ('dir1/file', b'file1-id'), ('dir2/', b'dir2-id'), ('dir2/file', b'file2-id')], basis=[('dir1/', b'dir-id'), ('dir1/file', b'file-id')], target=[('dir2/', b'dir-id'), ('dir2/file', b'file-id')])

    def test_invalid_file_not_present(self):
        state = self.assertBadDelta(active=[('file', b'file-id')], basis=[('file', b'file-id')], delta=[('other-file', 'file', b'file-id')])

    def test_invalid_new_id_same_path(self):
        state = self.assertBadDelta(active=[('file', b'file-id')], basis=[('file', b'file-id')], delta=[(None, 'file', b'file-id-2')])
        state = self.assertBadDelta(active=[('file', b'file-id-2')], basis=[('file', b'file-id-2')], delta=[(None, 'file', b'file-id')])

    def test_invalid_existing_id(self):
        state = self.assertBadDelta(active=[('file', b'file-id')], basis=[('file', b'file-id')], delta=[(None, 'file', b'file-id')])

    def test_invalid_parent_missing(self):
        state = self.assertBadDelta(active=[], basis=[], delta=[(None, 'path/path2', b'file-id')])
        state = self.assertBadDelta(active=[('path/', b'path-id')], basis=[], delta=[(None, 'path/path2', b'file-id')])
        state = self.assertBadDelta(active=[('path/', b'path-id'), ('path/path2', b'file-id')], basis=[], delta=[(None, 'path/path2', b'file-id')])

    def test_renamed_dir_same_path(self):
        state = self.assertUpdate(active=[('dir/', b'A-id'), ('dir/B', b'B-id')], basis=[('dir/', b'A-id'), ('dir/B', b'B-id')], target=[('dir/', b'C-id'), ('dir/B', b'B-id')])
        state = self.assertUpdate(active=[('dir/', b'C-id'), ('dir/B', b'B-id')], basis=[('dir/', b'A-id'), ('dir/B', b'B-id')], target=[('dir/', b'C-id'), ('dir/B', b'B-id')])
        state = self.assertUpdate(active=[], basis=[('dir/', b'A-id'), ('dir/B', b'B-id')], target=[('dir/', b'C-id'), ('dir/B', b'B-id')])
        state = self.assertUpdate(active=[('dir/', b'D-id'), ('dir/B', b'B-id')], basis=[('dir/', b'A-id'), ('dir/B', b'B-id')], target=[('dir/', b'C-id'), ('dir/B', b'B-id')])

    def test_parent_child_swap(self):
        state = self.assertUpdate(active=[('A/', b'A-id'), ('A/B/', b'B-id'), ('A/B/C', b'C-id')], basis=[('A/', b'A-id'), ('A/B/', b'B-id'), ('A/B/C', b'C-id')], target=[('A/', b'B-id'), ('A/B/', b'A-id'), ('A/B/C', b'C-id')])
        state = self.assertUpdate(active=[('A/', b'B-id'), ('A/B/', b'A-id'), ('A/B/C', b'C-id')], basis=[('A/', b'A-id'), ('A/B/', b'B-id'), ('A/B/C', b'C-id')], target=[('A/', b'B-id'), ('A/B/', b'A-id'), ('A/B/C', b'C-id')])
        state = self.assertUpdate(active=[], basis=[('A/', b'A-id'), ('A/B/', b'B-id'), ('A/B/C', b'C-id')], target=[('A/', b'B-id'), ('A/B/', b'A-id'), ('A/B/C', b'C-id')])
        state = self.assertUpdate(active=[('D/', b'A-id'), ('D/E/', b'B-id'), ('F', b'C-id')], basis=[('A/', b'A-id'), ('A/B/', b'B-id'), ('A/B/C', b'C-id')], target=[('A/', b'B-id'), ('A/B/', b'A-id'), ('A/B/C', b'C-id')])

    def test_change_root_id(self):
        state = self.assertUpdate(active=[('', b'root-id'), ('file', b'file-id')], basis=[('', b'root-id'), ('file', b'file-id')], target=[('', b'target-root-id'), ('file', b'file-id')])
        state = self.assertUpdate(active=[('', b'target-root-id'), ('file', b'file-id')], basis=[('', b'root-id'), ('file', b'file-id')], target=[('', b'target-root-id'), ('file', b'root-id')])
        state = self.assertUpdate(active=[('', b'active-root-id'), ('file', b'file-id')], basis=[('', b'root-id'), ('file', b'file-id')], target=[('', b'target-root-id'), ('file', b'root-id')])

    def test_change_file_absent_in_active(self):
        state = self.assertUpdate(active=[], basis=[('file', b'file-id')], target=[('file', b'file-id')])

    def test_invalid_changed_file(self):
        state = self.assertBadDelta(active=[('file', b'file-id')], basis=[], delta=[('file', 'file', b'file-id')])
        state = self.assertBadDelta(active=[('file', b'file-id')], basis=[('other-file', b'file-id')], delta=[('file', 'file', b'file-id')])