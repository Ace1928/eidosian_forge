from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
class TestInventoryFiltering(TestInventory):

    def test_inv_filter_empty(self):
        inv = self.prepare_inv_with_nested_dirs()
        new_inv = inv.filter([])
        self.assertEqual([('', b'tree-root')], [(path, ie.file_id) for path, ie in new_inv.iter_entries()])

    def test_inv_filter_files(self):
        inv = self.prepare_inv_with_nested_dirs()
        new_inv = inv.filter([b'zz-id', b'hello-id', b'a-id'])
        self.assertEqual([('', b'tree-root'), ('src', b'src-id'), ('src/hello.c', b'hello-id'), ('src/sub', b'sub-id'), ('src/sub/a', b'a-id'), ('zz', b'zz-id')], [(path, ie.file_id) for path, ie in new_inv.iter_entries()])

    def test_inv_filter_dirs(self):
        inv = self.prepare_inv_with_nested_dirs()
        new_inv = inv.filter([b'doc-id', b'sub-id'])
        self.assertEqual([('', b'tree-root'), ('doc', b'doc-id'), ('src', b'src-id'), ('src/sub', b'sub-id'), ('src/sub/a', b'a-id')], [(path, ie.file_id) for path, ie in new_inv.iter_entries()])

    def test_inv_filter_files_and_dirs(self):
        inv = self.prepare_inv_with_nested_dirs()
        new_inv = inv.filter([b'makefile-id', b'src-id'])
        self.assertEqual([('', b'tree-root'), ('Makefile', b'makefile-id'), ('src', b'src-id'), ('src/bye.c', b'bye-id'), ('src/hello.c', b'hello-id'), ('src/sub', b'sub-id'), ('src/sub/a', b'a-id'), ('src/zz.c', b'zzc-id')], [(path, ie.file_id) for path, ie in new_inv.iter_entries()])

    def test_inv_filter_entry_not_present(self):
        inv = self.prepare_inv_with_nested_dirs()
        new_inv = inv.filter([b'not-present-id'])
        self.assertEqual([('', b'tree-root')], [(path, ie.file_id) for path, ie in new_inv.iter_entries()])