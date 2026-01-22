from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
class TestCHKInventoryExpand(tests.TestCaseWithMemoryTransport):

    def get_chk_bytes(self):
        factory = groupcompress.make_pack_factory(True, True, 1)
        trans = self.get_transport('')
        return factory(trans)

    def make_dir(self, inv, name, parent_id, revision):
        ie = inv.make_entry('directory', name, parent_id, name.encode('utf-8') + b'-id')
        ie.revision = revision
        inv.add(ie)

    def make_file(self, inv, name, parent_id, revision, content=b'content\n'):
        ie = inv.make_entry('file', name, parent_id, name.encode('utf-8') + b'-id')
        ie.text_sha1 = osutils.sha_string(content)
        ie.text_size = len(content)
        ie.revision = revision
        inv.add(ie)

    def make_simple_inventory(self):
        inv = Inventory(b'TREE_ROOT')
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        self.make_dir(inv, 'dir1', b'TREE_ROOT', b'dirrev')
        self.make_dir(inv, 'dir2', b'TREE_ROOT', b'dirrev')
        self.make_dir(inv, 'sub-dir1', b'dir1-id', b'dirrev')
        self.make_file(inv, 'top', b'TREE_ROOT', b'filerev')
        self.make_file(inv, 'sub-file1', b'dir1-id', b'filerev')
        self.make_file(inv, 'sub-file2', b'dir1-id', b'filerev')
        self.make_file(inv, 'subsub-file1', b'sub-dir1-id', b'filerev')
        self.make_file(inv, 'sub2-file1', b'dir2-id', b'filerev')
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv, maximum_size=100, search_key_name=b'hash-255-way')
        lines = chk_inv.to_lines()
        return CHKInventory.deserialise(chk_bytes, lines, (b'revid',))

    def assert_Getitems(self, expected_fileids, inv, file_ids):
        self.assertEqual(sorted(expected_fileids), sorted([ie.file_id for ie in inv._getitems(file_ids)]))

    def assertExpand(self, all_ids, inv, file_ids):
        val_all_ids, val_children = inv._expand_fileids_to_parents_and_children(file_ids)
        self.assertEqual(set(all_ids), val_all_ids)
        entries = inv._getitems(val_all_ids)
        expected_children = {}
        for entry in entries:
            s = expected_children.setdefault(entry.parent_id, [])
            s.append(entry.file_id)
        val_children = {k: sorted(v) for k, v in val_children.items()}
        expected_children = {k: sorted(v) for k, v in expected_children.items()}
        self.assertEqual(expected_children, val_children)

    def test_make_simple_inventory(self):
        inv = self.make_simple_inventory()
        layout = []
        for path, entry in inv.iter_entries_by_dir():
            layout.append((path, entry.file_id))
        self.assertEqual([('', b'TREE_ROOT'), ('dir1', b'dir1-id'), ('dir2', b'dir2-id'), ('top', b'top-id'), ('dir1/sub-dir1', b'sub-dir1-id'), ('dir1/sub-file1', b'sub-file1-id'), ('dir1/sub-file2', b'sub-file2-id'), ('dir1/sub-dir1/subsub-file1', b'subsub-file1-id'), ('dir2/sub2-file1', b'sub2-file1-id')], layout)

    def test__getitems(self):
        inv = self.make_simple_inventory()
        self.assert_Getitems([b'dir1-id'], inv, [b'dir1-id'])
        self.assertTrue(b'dir1-id' in inv._fileid_to_entry_cache)
        self.assertFalse(b'sub-file2-id' in inv._fileid_to_entry_cache)
        self.assert_Getitems([b'dir1-id'], inv, [b'dir1-id'])
        self.assert_Getitems([b'dir1-id', b'sub-file2-id'], inv, [b'dir1-id', b'sub-file2-id'])
        self.assertTrue(b'dir1-id' in inv._fileid_to_entry_cache)
        self.assertTrue(b'sub-file2-id' in inv._fileid_to_entry_cache)

    def test_single_file(self):
        inv = self.make_simple_inventory()
        self.assertExpand([b'TREE_ROOT', b'top-id'], inv, [b'top-id'])

    def test_get_all_parents(self):
        inv = self.make_simple_inventory()
        self.assertExpand([b'TREE_ROOT', b'dir1-id', b'sub-dir1-id', b'subsub-file1-id'], inv, [b'subsub-file1-id'])

    def test_get_children(self):
        inv = self.make_simple_inventory()
        self.assertExpand([b'TREE_ROOT', b'dir1-id', b'sub-dir1-id', b'sub-file1-id', b'sub-file2-id', b'subsub-file1-id'], inv, [b'dir1-id'])

    def test_from_root(self):
        inv = self.make_simple_inventory()
        self.assertExpand([b'TREE_ROOT', b'dir1-id', b'dir2-id', b'sub-dir1-id', b'sub-file1-id', b'sub-file2-id', b'sub2-file1-id', b'subsub-file1-id', b'top-id'], inv, [b'TREE_ROOT'])

    def test_top_level_file(self):
        inv = self.make_simple_inventory()
        self.assertExpand([b'TREE_ROOT', b'top-id'], inv, [b'top-id'])

    def test_subsub_file(self):
        inv = self.make_simple_inventory()
        self.assertExpand([b'TREE_ROOT', b'dir1-id', b'sub-dir1-id', b'subsub-file1-id'], inv, [b'subsub-file1-id'])

    def test_sub_and_root(self):
        inv = self.make_simple_inventory()
        self.assertExpand([b'TREE_ROOT', b'dir1-id', b'sub-dir1-id', b'top-id', b'subsub-file1-id'], inv, [b'top-id', b'subsub-file1-id'])