from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
class TestInventoryReads(TestInventory):

    def test_is_root(self):
        """Ensure our root-checking code is accurate."""
        inv = self.make_init_inventory()
        self.assertTrue(inv.is_root(b'tree-root'))
        self.assertFalse(inv.is_root(b'booga'))
        ie = inv.get_entry(b'tree-root').copy()
        ie.file_id = b'booga'
        inv = inv.create_by_apply_delta([('', None, b'tree-root', None), (None, '', b'booga', ie)], b'new-rev-2')
        self.assertFalse(inv.is_root(b'TREE_ROOT'))
        self.assertTrue(inv.is_root(b'booga'))

    def test_ids(self):
        """Test detection of files within selected directories."""
        inv = inventory.Inventory(b'TREE_ROOT')
        inv.root.revision = b'revision'
        for args in [('src', 'directory', b'src-id'), ('doc', 'directory', b'doc-id'), ('src/hello.c', 'file'), ('src/bye.c', 'file', b'bye-id'), ('Makefile', 'file')]:
            ie = inv.add_path(*args)
            ie.revision = b'revision'
            if args[1] == 'file':
                ie.text_sha1 = osutils.sha_string(b'content\n')
                ie.text_size = len(b'content\n')
        inv = self.inv_to_test_inv(inv)
        self.assertEqual(inv.path2id('src'), b'src-id')
        self.assertEqual(inv.path2id('src/bye.c'), b'bye-id')

    def test_get_entry_by_path_partial(self):
        inv = inventory.Inventory(b'TREE_ROOT')
        inv.root.revision = b'revision'
        for args in [('src', 'directory', b'src-id'), ('doc', 'directory', b'doc-id'), ('src/hello.c', 'file'), ('src/bye.c', 'file', b'bye-id'), ('Makefile', 'file'), ('external', 'tree-reference', b'other-root')]:
            ie = inv.add_path(*args)
            ie.revision = b'revision'
            if args[1] == 'file':
                ie.text_sha1 = osutils.sha_string(b'content\n')
                ie.text_size = len(b'content\n')
            if args[1] == 'tree-reference':
                ie.reference_revision = b'reference'
        inv = self.inv_to_test_inv(inv)
        ie, resolved, remaining = inv.get_entry_by_path_partial('')
        self.assertEqual((ie.file_id, resolved, remaining), (b'TREE_ROOT', [], []))
        ie, resolved, remaining = inv.get_entry_by_path_partial('src')
        self.assertEqual((ie.file_id, resolved, remaining), (b'src-id', ['src'], []))
        ie, resolved, remaining = inv.get_entry_by_path_partial('src/bye.c')
        self.assertEqual((ie.file_id, resolved, remaining), (b'bye-id', ['src', 'bye.c'], []))
        ie, resolved, remaining = inv.get_entry_by_path_partial('external')
        self.assertEqual((ie.file_id, resolved, remaining), (b'other-root', ['external'], []))
        ie, resolved, remaining = inv.get_entry_by_path_partial('external/blah')
        self.assertEqual((ie.file_id, resolved, remaining), (b'other-root', ['external'], ['blah']))
        ie, resolved, remaining = inv.get_entry_by_path_partial('foo.c')
        self.assertEqual((ie, resolved, remaining), (None, None, None))

    def test_non_directory_children(self):
        """Test path2id when a parent directory has no children"""
        inv = inventory.Inventory(b'tree-root')
        inv.add(self.make_file(b'file-id', 'file', b'tree-root'))
        inv.add(self.make_link(b'link-id', 'link', b'tree-root'))
        self.assertIs(None, inv.path2id('file/subfile'))
        self.assertIs(None, inv.path2id('link/subfile'))

    def test_is_unmodified(self):
        f1 = self.make_file(b'file-id', 'file', b'tree-root')
        f1.revision = b'rev'
        self.assertTrue(f1.is_unmodified(f1))
        f2 = self.make_file(b'file-id', 'file', b'tree-root')
        f2.revision = b'rev'
        self.assertTrue(f1.is_unmodified(f2))
        f3 = self.make_file(b'file-id', 'file', b'tree-root')
        self.assertFalse(f1.is_unmodified(f3))
        f4 = self.make_file(b'file-id', 'file', b'tree-root')
        f4.revision = b'rev1'
        self.assertFalse(f1.is_unmodified(f4))

    def test_iter_entries(self):
        inv = self.prepare_inv_with_nested_dirs()
        self.assertEqual([('', b'tree-root'), ('Makefile', b'makefile-id'), ('doc', b'doc-id'), ('src', b'src-id'), ('src/bye.c', b'bye-id'), ('src/hello.c', b'hello-id'), ('src/sub', b'sub-id'), ('src/sub/a', b'a-id'), ('src/zz.c', b'zzc-id'), ('zz', b'zz-id')], [(path, ie.file_id) for path, ie in inv.iter_entries()])
        self.assertEqual([('bye.c', b'bye-id'), ('hello.c', b'hello-id'), ('sub', b'sub-id'), ('sub/a', b'a-id'), ('zz.c', b'zzc-id')], [(path, ie.file_id) for path, ie in inv.iter_entries(from_dir=b'src-id')])
        self.assertEqual([('', b'tree-root'), ('Makefile', b'makefile-id'), ('doc', b'doc-id'), ('src', b'src-id'), ('zz', b'zz-id')], [(path, ie.file_id) for path, ie in inv.iter_entries(recursive=False)])
        self.assertEqual([('bye.c', b'bye-id'), ('hello.c', b'hello-id'), ('sub', b'sub-id'), ('zz.c', b'zzc-id')], [(path, ie.file_id) for path, ie in inv.iter_entries(from_dir=b'src-id', recursive=False)])

    def test_iter_just_entries(self):
        inv = self.prepare_inv_with_nested_dirs()
        self.assertEqual([b'a-id', b'bye-id', b'doc-id', b'hello-id', b'makefile-id', b'src-id', b'sub-id', b'tree-root', b'zz-id', b'zzc-id'], sorted([ie.file_id for ie in inv.iter_just_entries()]))

    def test_iter_entries_by_dir(self):
        inv = self.prepare_inv_with_nested_dirs()
        self.assertEqual([('', b'tree-root'), ('Makefile', b'makefile-id'), ('doc', b'doc-id'), ('src', b'src-id'), ('zz', b'zz-id'), ('src/bye.c', b'bye-id'), ('src/hello.c', b'hello-id'), ('src/sub', b'sub-id'), ('src/zz.c', b'zzc-id'), ('src/sub/a', b'a-id')], [(path, ie.file_id) for path, ie in inv.iter_entries_by_dir()])
        self.assertEqual([('', b'tree-root'), ('Makefile', b'makefile-id'), ('doc', b'doc-id'), ('src', b'src-id'), ('zz', b'zz-id'), ('src/bye.c', b'bye-id'), ('src/hello.c', b'hello-id'), ('src/sub', b'sub-id'), ('src/zz.c', b'zzc-id'), ('src/sub/a', b'a-id')], [(path, ie.file_id) for path, ie in inv.iter_entries_by_dir(specific_file_ids=(b'a-id', b'zzc-id', b'doc-id', b'tree-root', b'hello-id', b'bye-id', b'zz-id', b'src-id', b'makefile-id', b'sub-id'))])
        self.assertEqual([('Makefile', b'makefile-id'), ('doc', b'doc-id'), ('zz', b'zz-id'), ('src/bye.c', b'bye-id'), ('src/hello.c', b'hello-id'), ('src/zz.c', b'zzc-id'), ('src/sub/a', b'a-id')], [(path, ie.file_id) for path, ie in inv.iter_entries_by_dir(specific_file_ids=(b'a-id', b'zzc-id', b'doc-id', b'hello-id', b'bye-id', b'zz-id', b'makefile-id'))])
        self.assertEqual([('Makefile', b'makefile-id'), ('src/bye.c', b'bye-id')], [(path, ie.file_id) for path, ie in inv.iter_entries_by_dir(specific_file_ids=(b'bye-id', b'makefile-id'))])
        self.assertEqual([('Makefile', b'makefile-id'), ('src/bye.c', b'bye-id')], [(path, ie.file_id) for path, ie in inv.iter_entries_by_dir(specific_file_ids=(b'bye-id', b'makefile-id'))])
        self.assertEqual([('src/bye.c', b'bye-id')], [(path, ie.file_id) for path, ie in inv.iter_entries_by_dir(specific_file_ids=(b'bye-id',))])